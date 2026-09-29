//! Naive reference ops for M1 alignment (no SIMD yet).
//! Layout: channel-first [C, T], matching QORA conv.rs convention.

/// Fuse weight_norm: w = g * v / ||v||  (per out-channel).
/// g: [out], v: [out, in, k] flattened per out-channel.
pub fn fuse_weight_norm(g: &[f32], v: &[f32], out_c: usize, in_c: usize, k: usize) -> Vec<f32> {
    assert_eq!(g.len(), out_c);
    assert_eq!(v.len(), out_c * in_c * k);
    let mut w = vec![0.0f32; v.len()];
    let norm_len = in_c * k;
    for o in 0..out_c {
        let base = o * norm_len;
        let mut s = 0.0f64;
        for i in 0..norm_len {
            let x = v[base + i] as f64;
            s += x * x;
        }
        let n = s.sqrt().max(1e-12);
        let scale = g[o] as f64 / n;
        for i in 0..norm_len {
            w[base + i] = (v[base + i] as f64 * scale) as f32;
        }
    }
    w
}

/// Causal Conv1d, stride=1: left-pad (pad*2) then valid conv.
pub fn causal_conv1d(x: &[f32], c_in: usize, t_in: usize, w: &[f32], c_out: usize, k: usize, pad: usize) -> Vec<f32> {
    let t_pad = t_in + pad * 2;
    let mut out = vec![0.0f32; c_out * t_in];
    for o in 0..c_out {
        for t in 0..t_in {
            let mut acc = 0.0f32;
            for c in 0..c_in {
                for kk in 0..k {
                    let tp = t + pad * 2 + 1 - k + kk; // index into padded (left pad only on original)
                    // padded[t] = 0 if t < pad*2 else x[t - pad*2]
                    let v = if tp < pad * 2 { 0.0 } else { x[c * t_in + (tp - pad * 2)] };
                    acc += v * w[(o * c_in + c) * k + kk];
                }
            }
            out[o * t_in + t] = acc;
        }
    }
    let _ = t_pad;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fuse_matches_torch() {
        // torch.nn.utils.weight_norm: norm over (in,k) per out
        let g = vec![2.0f32];
        let v = vec![3.0f32, 4.0];
        let w = fuse_weight_norm(&g, &v, 1, 1, 2);
        // norm=5, w = 2*[0.6,0.8]
        assert!((w[0] - 1.2).abs() < 1e-6);
        assert!((w[1] - 1.6).abs() < 1e-6);
    }
}
