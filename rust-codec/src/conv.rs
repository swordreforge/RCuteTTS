//! Naive reference ops for M1 alignment (no SIMD yet).
//! Layout: channel-first, flat `[C, T]` row-major, matching QORA conv.rs.
//!
//! Semantics mirror `src/cutetts/audio_codec/model/audio_vae.py`:
//! - `CausalConv1d`: left-pad `(pad*2, 0)`, then plain Conv1d (padding 0).
//!   `weight_norm` is fused offline via [`fuse_weight_norm`].
//! - `CausalTransposeConv1d`: plain ConvTranspose1d (padding 0), then trim
//!   `trim = pad*2 - out_pad` samples off the right.

/// Fuse `torch.nn.utils.weight_norm` (dim=0): `w = g * v / ||v||` per out-channel.
/// `g: [out_c]`, `v: [out_c, in_c_per_group, k]` flattened.
pub fn fuse_weight_norm(g: &[f32], v: &[f32], out_c: usize, in_c_g: usize, k: usize) -> Vec<f32> {
    assert_eq!(g.len(), out_c);
    assert_eq!(v.len(), out_c * in_c_g * k);
    let mut w = vec![0.0f32; v.len()];
    let norm_len = in_c_g * k;
    for o in 0..out_c {
        let base = o * norm_len;
        let mut s = 0.0f64;
        for i in 0..norm_len {
            let x = v[base + i] as f64;
            s += x * x;
        }
        let scale = g[o] as f64 / s.sqrt().max(1e-12);
        for i in 0..norm_len {
            w[base + i] = (v[base + i] as f64 * scale) as f32;
        }
    }
    w
}

/// Grouped causal Conv1d, stride=1.
/// `x: [c_in, t]`, `w: [c_out, c_in/groups, k]` (already fused), `bias: [c_out]`.
/// `pad` = the module's `padding` arg (left pad is `pad*2`).
pub fn causal_conv1d(
    x: &[f32],
    c_in: usize,
    t: usize,
    w: &[f32],
    bias: &[f32],
    c_out: usize,
    k: usize,
    dilation: usize,
    groups: usize,
    pad: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), c_in * t);
    assert_eq!(bias.len(), c_out);
    assert_eq!(c_in % groups, 0);
    assert_eq!(c_out % groups, 0);
    let cig = c_in / groups;
    assert_eq!(w.len(), c_out * cig * k);
    let left = pad * 2;
    let mut out = vec![0.0f32; c_out * t];
    for o in 0..c_out {
        let g = o / (c_out / groups);
        let c0 = g * cig;
        for ti in 0..t {
            let mut acc = bias[o] as f64;
            for ci in 0..cig {
                for kk in 0..k {
                    // index into left-padded input
                    let p = ti as isize + (kk * dilation) as isize - left as isize;
                    // padded[p'] where p' = p + left; value 0 if p' < left
                    let src = p + left as isize;
                    let v = if src < left as isize {
                        0.0
                    } else {
                        x[(c0 + ci) * t + (src - left as isize) as usize]
                    };
                    acc += v as f64 * w[(o * cig + ci) * k + kk] as f64;
                }
            }
            out[o * t + ti] = acc as f32;
        }
    }
    out
}

/// Grouped causal ConvTranspose1d, matching `CausalTransposeConv1d`:
/// full transpose (stride `s`, padding 0), then drop `trim = pad*2 - out_pad`
/// samples on the right. `w: [c_in, c_out/groups, k]` (ConvTranspose layout).
pub fn causal_transpose_conv1d(
    x: &[f32],
    c_in: usize,
    t: usize,
    w: &[f32],
    bias: &[f32],
    c_out: usize,
    k: usize,
    stride: usize,
    groups: usize,
    pad: usize,
    out_pad: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), c_in * t);
    assert_eq!(bias.len(), c_out);
    let cog = c_out / groups;
    assert_eq!(w.len(), c_in * cog * k);
    let full_t = (t - 1) * stride + k;
    let trim = pad * 2 - out_pad;
    assert!(trim < full_t);
    let out_t = full_t - trim;
    let mut full = vec![0.0f32; c_out * full_t];
    for (c, xi) in x.chunks_exact(t).enumerate() {
        let g = c / (c_in / groups);
        for (i, &xv) in xi.iter().enumerate() {
            if xv == 0.0 {
                continue;
            }
            for og in 0..cog {
                let o = g * cog + og;
                let wbase = (c * cog + og) * k;
                let obase = o * full_t + i * stride;
                for kk in 0..k {
                    full[obase + kk] += xv * w[wbase + kk];
                }
            }
        }
    }
    let mut out = vec![0.0f32; c_out * out_t];
    for o in 0..c_out {
        let b = bias[o];
        for ti in 0..out_t {
            out[o * out_t + ti] = full[o * full_t + ti] + b;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fuse_matches_torch() {
        let g = vec![2.0f32];
        let v = vec![3.0f32, 4.0];
        let w = fuse_weight_norm(&g, &v, 1, 1, 2);
        assert!((w[0] - 1.2).abs() < 1e-6);
        assert!((w[1] - 1.6).abs() < 1e-6);
    }

    #[test]
    fn conv1x1_identity() {
        let x = vec![1.0f32, 2.0, 3.0];
        let w = vec![1.0f32];
        let y = causal_conv1d(&x, 1, 3, &w, &[0.0], 1, 1, 1, 1, 0);
        assert_eq!(y, x);
    }

    #[test]
    fn transpose_stride1_is_conv_flipped() {
        // stride=1 transpose == true convolution; with a symmetric kernel it
        // must equal causal conv k=3 pad=1 (trim=2).
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![0.5f32, 1.0, 0.5];
        let yt = causal_transpose_conv1d(&x, 1, 4, &w, &[0.0], 1, 3, 1, 1, 1, 0);
        let yc = causal_conv1d(&x, 1, 4, &w, &[0.0], 1, 3, 1, 1, 1);
        assert_eq!(yt.len(), yc.len());
        for (a, b) in yt.iter().zip(yc.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }
}
