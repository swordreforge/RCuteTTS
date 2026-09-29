//! Snake1d: x + sin^2(alpha * x) / (alpha + 1e-9), per-channel alpha [C].
//! Matches `snake()` in src/cutetts/audio_codec/model/audio_vae.py:73-79.

pub fn snake1d(x: &mut [f32], channels: usize, t: usize, alpha: &[f32]) {
    assert_eq!(alpha.len(), channels);
    assert_eq!(x.len(), channels * t);
    for c in 0..channels {
        let a = alpha[c];
        let inv = 1.0 / (a + 1e-9);
        for i in 0..t {
            let idx = c * t + i;
            let v = x[idx];
            let s = (a * v).sin();
            x[idx] = v + inv * s * s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn snake_alpha_one() {
        let mut x = vec![0.0f32, 1.0];
        snake1d(&mut x, 1, 2, &[1.0]);
        assert!((x[0] - 0.0).abs() < 1e-6);
        let expect = 1.0 + (1.0f32.sin().powi(2));
        assert!((x[1] - expect).abs() < 1e-6);
    }
}
