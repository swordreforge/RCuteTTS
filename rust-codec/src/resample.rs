//! sinc_interp_hann resampling (torchaudio-compatible defaults).
//!
//! Mirrors `torchaudio.functional.resample` (width=6, rolloff=0.99):
//! kernel built in f64 then cast to f32, zero padding (width left,
//! width+orig right), polyphase conv stride=orig, output truncated to
//! ceil(new*L/orig). orig==new is a passthrough (clone).
//! Output j = n*new+r: sum_kk ker[r,kk] * padded[n*orig+kk].

use std::f64::consts::PI;

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Sinc-resample kernel rows [new, K] (f32), K = 2*width+orig.
fn sinc_kernel(orig: usize, new: usize, width: usize, base_freq: f64) -> Vec<f32> {
    let k = 2 * width + orig;
    let mut ker = vec![0.0f32; new * k];
    for r in 0..new {
        for (kk, dst) in ker[r * k..(r + 1) * k].iter_mut().enumerate() {
            // idx = (kk - width) / orig  [mirrors arange(-width, width+orig)/orig]
            // t = -r/new + idx, scaled by base_freq, clamped
            let mut t = -(r as f64) / new as f64 + (kk as f64 - width as f64) / orig as f64;
            t *= base_freq;
            t = t.clamp(-6.0, 6.0);
            let window = (t * PI / 6.0 / 2.0).cos().powi(2);
            let v = t * PI;
            let s = if v == 0.0 { 1.0 } else { v.sin() / v };
            // scale = base_freq / orig
            *dst = (s * window * (base_freq / orig as f64)) as f32;
        }
    }
    ker
}

/// Resample mono `wave` from `orig_freq` to `new_freq`.
pub fn resample(wave: &[f32], orig_freq: usize, new_freq: usize) -> Vec<f32> {
    assert!(orig_freq > 0 && new_freq > 0);
    if orig_freq == new_freq {
        return wave.to_vec();
    }
    let g = gcd(orig_freq, new_freq);
    let (orig, new) = (orig_freq / g, new_freq / g);
    let base_freq = (orig.min(new) as f64) * 0.99;
    let width = ((6.0 * orig as f64 / base_freq).ceil()) as usize;
    let ker = sinc_kernel(orig, new, width, base_freq);
    let k = 2 * width + orig;
    // zero pad: width left, width+orig right
    let mut padded = vec![0.0f32; wave.len() + 2 * width + orig];
    padded[width..width + wave.len()].copy_from_slice(wave);
    let n_full = (padded.len() - k) / orig + 1;
    let target = (new * wave.len() + orig - 1) / orig; // ceil
    let mut out = vec![0.0f32; n_full * new];
    for n in 0..n_full {
        for r in 0..new {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += ker[r * k + kk] * padded[n * orig + kk];
            }
            out[n * new + r] = acc;
        }
    }
    out.truncate(target);
    out
}
