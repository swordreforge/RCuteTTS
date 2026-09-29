//! Full-sentence AudioVAE decode in Rust.
//!
//! Mirrors `CausalDecoder.forward` (`audio_vae.py:243-292`):
//! front depthwise k7 + 1x1, then per stage
//! `Snake -> Transpose(k=2s,s) -> 3x ResUnit(d=1,3,9)`, finally
//! `Snake -> Conv k7 -> Tanh`.

use crate::conv::{causal_conv1d, causal_transpose_conv1d};
use crate::snake::snake1d;
use crate::weights::DecoderW;

const RES_DILATION: [usize; 3] = [1, 3, 9];
const RES_PAD: [usize; 3] = [3, 9, 27];

/// `latent: [64, frames]` -> mono waveform (`frames * 1920` samples).
pub fn decode(w: &DecoderW, latent: &[f32], frames: usize) -> Vec<f32> {
    assert_eq!(latent.len(), 64 * frames);
    let mut h = causal_conv1d(latent, 64, frames, &w.front_dw_w, &w.front_dw_b, 64, 7, 1, 64, 3);
    h = causal_conv1d(&h, 64, frames, &w.front_pw_w, &w.front_pw_b, 1536, 1, 1, 1, 0);
    let (mut t, mut ch) = (frames, 1536);
    for st in &w.stages {
        snake1d(&mut h, ch, t, &st.alpha);
        let pad = (st.stride + 1) / 2; // ceil(s/2)
        h = causal_transpose_conv1d(
            &h, ch, t, &st.trans_w, &st.trans_b, st.out_c, 2 * st.stride, st.stride, 1, pad,
            st.stride % 2,
        );
        t *= st.stride;
        ch = st.out_c;
        for (ri, ru) in st.res.iter().enumerate() {
            let mut r = h.clone();
            snake1d(&mut r, ch, t, &ru.alpha0);
            r = causal_conv1d(&r, ch, t, &ru.dw_w, &ru.dw_b, ch, 7, RES_DILATION[ri], ch, RES_PAD[ri]);
            snake1d(&mut r, ch, t, &ru.alpha1);
            r = causal_conv1d(&r, ch, t, &ru.pw_w, &ru.pw_b, ch, 1, 1, 1, 0);
            h = h.iter().zip(r.iter()).map(|(a, b)| a + b).collect();
        }
        let _ = st.in_c;
    }
    snake1d(&mut h, 96, t, &w.final_alpha);
    h = causal_conv1d(&h, 96, t, &w.final_w, &w.final_b, 1, 7, 1, 1, 3);
    h.iter().map(|v| v.tanh()).collect()
}
