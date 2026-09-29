//! Snake1d: x + sin^2(alpha * x) / (alpha + 1e-9), per-channel alpha [C].
//! Matches `snake()` in src/cutetts/audio_codec/model/audio_vae.py:73-79.

pub fn snake1d(x: &mut [f32], channels: usize, t: usize, alpha: &[f32]) {
    snake1d_nth(x, channels, t, alpha, 1);
}

/// Threaded [`snake1d`] over channels. Bit-identical to serial.
pub fn snake1d_nth(x: &mut [f32], channels: usize, t: usize, alpha: &[f32], n_threads: usize) {
    assert_eq!(alpha.len(), channels);
    assert_eq!(x.len(), channels * t);
    if n_threads <= 1 || channels <= 1 || x.len() < 200_000 {
        return snake_rows(x, channels, t, alpha, 0, channels);
    }
    let per = (channels + n_threads - 1) / n_threads;
    std::thread::scope(|s| {
        for (ci, chunk) in x.chunks_mut(per * t).enumerate() {
            let c0 = ci * per;
            s.spawn(move || {
                snake_rows(chunk, channels, t, alpha, c0, (c0 + per).min(channels));
            });
        }
    });
}

fn snake_rows(x: &mut [f32], _channels: usize, t: usize, alpha: &[f32], c0: usize, c1: usize) {
    for c in c0..c1 {
        let a = alpha[c];
        let inv = 1.0 / (a + 1e-9);
        let row = &mut x[(c - c0) * t..(c - c0 + 1) * t];
        for v in row.iter_mut() {
            let s = (*v * a).sin();
            *v += inv * s * s;
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
