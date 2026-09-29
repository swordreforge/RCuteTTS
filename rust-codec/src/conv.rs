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
/// f32 accumulation (matches torch numerics within 1e-6, auto-vectorizes).
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
    let mut out = vec![0.0f32; c_out * t];
    conv_rows(x, w, bias, c_in, c_out, t, k, dilation, groups, pad, 0, c_out, &mut out);
    out
}

fn conv_rows(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    c_in: usize,
    c_out: usize,
    t: usize,
    k: usize,
    dilation: usize,
    groups: usize,
    pad: usize,
    o0: usize,
    o1: usize,
    out: &mut [f32],
) {
    let cig = c_in / groups;
    let left = pad * 2;
    for o in o0..o1 {
        let g = o / (c_out / groups);
        let c0 = g * cig;
        let orow = &mut out[(o - o0) * t..(o - o0 + 1) * t];
        if k == 1 && dilation == 1 && pad == 0 {
            // Pure 1x1 GEMM row: SAXPY form, contiguous + FMA-friendly.
            orow.fill(bias[o]);
            for ci in 0..cig {
                let s = w[o * cig + ci];
                let xrow = &x[(c0 + ci) * t..(c0 + ci + 1) * t];
                for ti in 0..t {
                    orow[ti] += xrow[ti] * s;
                }
            }
            continue;
        }
        // ti >= left: every tap is in range -> branch-free, auto-vectorizable.
        // ti < left (at most a few dozen): bounds-checked slow path.
        let fast_from = left.min(t);
        for ti in 0..fast_from {
            let mut acc = bias[o];
            for ci in 0..cig {
                let xrow = &x[(c0 + ci) * t..(c0 + ci + 1) * t];
                let wrow = &w[(o * cig + ci) * k..(o * cig + ci + 1) * k];
                for kk in 0..k {
                    let src = ti as isize + (kk * dilation) as isize - left as isize;
                    if src >= 0 {
                        acc += xrow[src as usize] * wrow[kk];
                    }
                }
            }
            orow[ti] = acc;
        }
        for ti in fast_from..t {
            let mut acc = bias[o];
            for ci in 0..cig {
                let base = (c0 + ci) * t + ti - left;
                let wbase = (o * cig + ci) * k;
                for kk in 0..k {
                    acc += x[base + kk * dilation] * w[wbase + kk];
                }
            }
            orow[ti] = acc;
        }
    }
}

/// Threaded [`causal_conv1d`]: split out-channels over `n_threads` workers.
/// Partitioning is by whole output rows, so results are bit-identical to serial.
pub fn causal_conv1d_par(
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
    n_threads: usize,
) -> Vec<f32> {
    let ops = c_out as u64 * t as u64 * (c_in / groups) as u64 * k as u64;
    if n_threads <= 1 || c_out <= 1 || ops < 200_000 {
        return causal_conv1d(x, c_in, t, w, bias, c_out, k, dilation, groups, pad);
    }
    let per = (c_out + n_threads - 1) / n_threads;
    let mut out = vec![0.0f32; c_out * t];
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per * t).enumerate() {
            let o0 = ci * per;
            let o1 = (o0 + per).min(c_out);
            s.spawn(move || {
                conv_rows(x, w, bias, c_in, c_out, t, k, dilation, groups, pad, o0, o1, chunk);
            });
        }
    });
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
    transpose_rows(x, w, c_in, c_out, t, k, stride, groups, full_t, 0, c_out, &mut full);
    let mut out = vec![0.0f32; c_out * out_t];
    for o in 0..c_out {
        let b = bias[o];
        for ti in 0..out_t {
            out[o * out_t + ti] = full[o * full_t + ti] + b;
        }
    }
    out
}

fn transpose_rows(
    x: &[f32],
    w: &[f32],
    c_in: usize,
    c_out: usize,
    t: usize,
    k: usize,
    stride: usize,
    groups: usize,
    full_t: usize,
    o0: usize,
    o1: usize,
    full: &mut [f32],
) {
    let cog = c_out / groups;
    let cig = c_in / groups;
    for o in o0..o1 {
        let g = o / cog;
        let frow = &mut full[(o - o0) * full_t..(o - o0 + 1) * full_t];
        let og = o % cog;
        for cg in 0..cig {
            let c = g * cig + cg;
            let xi = &x[c * t..(c + 1) * t];
            let wrow = &w[(c * cog + og) * k..(c * cog + og + 1) * k];
            for (i, &xv) in xi.iter().enumerate() {
                if xv == 0.0 {
                    continue;
                }
                let base = i * stride;
                for kk in 0..k {
                    frow[base + kk] += xv * wrow[kk];
                }
            }
        }
    }
}

/// Threaded [`causal_transpose_conv1d`]: split out-channels over workers.
/// Bit-identical to serial (whole output rows per worker).
pub fn causal_transpose_conv1d_par(
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
    n_threads: usize,
) -> Vec<f32> {
    let ops = c_out as u64 * t as u64 * stride as u64 * (c_in / groups.max(1)) as u64 * k as u64;
    if n_threads <= 1 || c_out <= 1 || ops < 200_000 {
        return causal_transpose_conv1d(x, c_in, t, w, bias, c_out, k, stride, groups, pad, out_pad);
    }
    let full_t = (t - 1) * stride + k;
    let trim = pad * 2 - out_pad;
    let out_t = full_t - trim;
    let per = (c_out + n_threads - 1) / n_threads;
    let mut full = vec![0.0f32; c_out * full_t];
    std::thread::scope(|s| {
        for (ci, chunk) in full.chunks_mut(per * full_t).enumerate() {
            let o0 = ci * per;
            let o1 = (o0 + per).min(c_out);
            s.spawn(move || {
                transpose_rows(x, w, c_in, c_out, t, k, stride, groups, full_t, o0, o1, chunk);
            });
        }
    });
    let mut out = vec![0.0f32; c_out * out_t];
    for o in 0..c_out {
        let b = bias[o];
        for ti in 0..out_t {
            out[o * out_t + ti] = full[o * full_t + ti] + b;
        }
    }
    out
}

/// Worker count: `CUTETTS_THREADS` override, else all logical CPUs.
pub fn default_threads() -> usize {
    if let Ok(v) = std::env::var("CUTETTS_THREADS") {
        if let Ok(n) = v.parse::<usize>() {
            if n >= 1 {
                return n;
            }
        }
    }
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
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
