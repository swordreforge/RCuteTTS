//! Full-sentence AudioVAE decode in Rust.
//!
//! Mirrors `CausalDecoder.forward` (`audio_vae.py:243-292`):
//! front depthwise k7 + 1x1, then per stage
//! `Snake -> Transpose(k=2s,s) -> 3x ResUnit(d=1,3,9)`, finally
//! `Snake -> Conv k7 -> Tanh`.

use crate::conv::{causal_conv1d_par, causal_transpose_conv1d_par, default_threads};
use crate::snake::snake1d_nth as snake1d_par;
use crate::weights::DecoderW;

const RES_DILATION: [usize; 3] = [1, 3, 9];
const RES_PAD: [usize; 3] = [3, 9, 27];

fn cv(x: &[f32], c_in: usize, t: usize, w: &[f32], b: &[f32], c_out: usize, k: usize,
      d: usize, g: usize, p: usize, nth: usize) -> Vec<f32> {
    causal_conv1d_par(x, c_in, t, w, b, c_out, k, d, g, p, nth)
}

fn tc(x: &[f32], c_in: usize, t: usize, w: &[f32], b: &[f32], c_out: usize, k: usize,
      s: usize, pad: usize, out_pad: usize, nth: usize) -> Vec<f32> {
    causal_transpose_conv1d_par(x, c_in, t, w, b, c_out, k, s, 1, pad, out_pad, nth)
}

/// `latent: [64, frames]` -> mono waveform (`frames * 1920` samples).
pub fn decode(w: &DecoderW, latent: &[f32], frames: usize) -> Vec<f32> {
    decode_nth(w, latent, frames, default_threads())
}

pub fn decode_nth(w: &DecoderW, latent: &[f32], frames: usize, nth: usize) -> Vec<f32> {
    assert_eq!(latent.len(), 64 * frames);
    let mut h = cv(latent, 64, frames, &w.front_dw_w, &w.front_dw_b, 64, 7, 1, 64, 3, nth);
    h = cv(&h, 64, frames, &w.front_pw_w, &w.front_pw_b, 1536, 1, 1, 1, 0, nth);
    let (mut t, mut ch) = (frames, 1536);
    for st in &w.stages {
        snake1d_par(&mut h, ch, t, &st.alpha, nth);
        let pad = (st.stride + 1) / 2; // ceil(s/2)
        h = tc(&h, ch, t, &st.trans_w, &st.trans_b, st.out_c, 2 * st.stride,
               st.stride, pad, st.stride % 2, nth);
        t *= st.stride;
        ch = st.out_c;
        for (ri, ru) in st.res.iter().enumerate() {
            let mut r = h.clone();
            snake1d_par(&mut r, ch, t, &ru.alpha0, nth);
            r = cv(&r, ch, t, &ru.dw_w, &ru.dw_b, ch, 7, RES_DILATION[ri], ch, RES_PAD[ri], nth);
            snake1d_par(&mut r, ch, t, &ru.alpha1, nth);
            r = cv(&r, ch, t, &ru.pw_w, &ru.pw_b, ch, 1, 1, 1, 0, nth);
            h = h.iter().zip(r.iter()).map(|(a, b)| a + b).collect();
        }
        let _ = st.in_c;
    }
    snake1d_par(&mut h, 96, t, &w.final_alpha, nth);
    h = cv(&h, 96, t, &w.final_w, &w.final_b, 1, 7, 1, 1, 3, nth);
    h.iter().map(|v| v.tanh()).collect()
}
