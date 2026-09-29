//! Full-sentence AudioVAE decode in Rust.
//!
//! Mirrors `CausalDecoder.forward` (`audio_vae.py:243-292`):
//! front depthwise k7 + 1x1, then per stage
//! `Snake -> Transpose(k=2s,s) -> 3x ResUnit(d=1,3,9)`, finally
//! `Snake -> Conv k7 -> Tanh`.

use crate::conv::{causal_conv1d_par, default_threads};
use crate::gemm::sgemm_bias;
use std::time::Instant;

fn profile() -> bool {
    std::env::var("CUTETTS_PROFILE").map(|v| v == "1").unwrap_or(false)
}

macro_rules! pt {
    ($prof:expr, $t0:expr, $name:expr, $block:block) => {{
        let __t0 = $t0;
        let r = $block;
        if $prof {
            eprintln!("[prof] {:24} {:8.1}ms", $name, __t0.elapsed().as_secs_f32() * 1000.0);
        }
        r
    }};
}
use crate::snake::snake1d_nth as snake1d_par;
use crate::weights::DecoderW;

pub(crate) const RES_DILATION: [usize; 3] = [1, 3, 9];
pub(crate) const RES_PAD: [usize; 3] = [3, 9, 27];

pub(crate) fn cv(x: &[f32], c_in: usize, t: usize, w: &[f32], b: &[f32], c_out: usize, k: usize,
      d: usize, g: usize, p: usize, nth: usize) -> Vec<f32> {
    causal_conv1d_par(x, c_in, t, w, b, c_out, k, d, g, p, nth)
}

/// Full transposed-conv + interleave over T inputs (no slicing).
/// Streaming reuses this on [history ++ chunk] and slices rows [s..].
pub(crate) fn transpose_full(x: &[f32], t: usize, st: &crate::weights::StageW, nth: usize) -> Vec<f32> {
    // Transpose k=2s exact decomposition:
    // out[o, i*s+j] = Y0[j][o,i] + (i>0 ? Y1[j][o,i-1] : 0), bias folded into Y0.
    // Y = trans_gemm @ x, rows [M0_0..M0_{s-1}, M1_0..M1_{s-1}].
    let (s, o) = (st.stride, st.out_c);
    let mut y = vec![0.0f32; st.trans_gemm.rows * t];
    sgemm_bias(&st.trans_gemm, x, t, &st.trans_bias_pad, &mut y, nth);
    let mut out = vec![0.0f32; o * t * s];
    let ops = o as u64 * t as u64 * s as u64;
    if nth <= 1 || o <= 1 || ops < 500_000 {
        interleave(&y, &mut out, o, t, s);
    } else {
        let per = (o + nth - 1) / nth;
        std::thread::scope(|scope| {
            for (ci, chunk) in out.chunks_mut(per * t * s).enumerate() {
                let o0 = ci * per;
                let o1 = (o0 + per).min(o);
                let yref = &y;
                scope.spawn(move || {
                    interleave_range(yref, chunk, o, t, s, o0, o1);
                });
            }
        });
    }
    out
}

fn interleave(y: &[f32], out: &mut [f32], o: usize, t: usize, s: usize) {
    interleave_range(y, out, o, t, s, 0, o);
}

fn interleave_range(y: &[f32], out: &mut [f32], o: usize, t: usize, s: usize, o0: usize, o1: usize) {
    for oo in o0..o1 {
        for i in 0..t {
            for j in 0..s {
                let y0 = y[(j * o + oo) * t + i];
                let y1 = if i > 0 { y[((s + j) * o + oo) * t + i - 1] } else { 0.0 };
                out[((oo - o0) * t + i) * s + j] = y0 + y1;
            }
        }
    }
}

pub(crate) fn pw1(x: &[f32], t: usize, g: &crate::gemm::PackedA, b: &[f32], o: usize, nth: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; g.rows * t];
    sgemm_bias(g, x, t, b, &mut c, nth);
    c.truncate(o * t);
    c
}

/// `latent: [64, frames]` -> mono waveform (`frames * 1920` samples).
pub fn decode(w: &DecoderW, latent: &[f32], frames: usize) -> Vec<f32> {
    decode_nth(w, latent, frames, default_threads())
}

pub fn decode_nth(w: &DecoderW, latent: &[f32], frames: usize, nth: usize) -> Vec<f32> {
    assert_eq!(latent.len(), 64 * frames);
    let prof = profile();
    let mut h = pt!(prof, Instant::now(), "front_dw", {
        cv(latent, 64, frames, &w.front_dw_w, &w.front_dw_b, 64, 7, 1, 64, 3, nth)
    });
    h = pt!(prof, Instant::now(), "front_pw", {
        pw1(&h, frames, &w.front_pw_gemm, &w.front_pw_bias_pad, 1536, nth)
    });
    let (mut t, mut ch) = (frames, 1536);
    for (si, st) in w.stages.iter().enumerate() {
        pt!(prof, Instant::now(), format!("s{si}_snake0"), {
            snake1d_par(&mut h, ch, t, &st.alpha, nth)
        });
        h = pt!(prof, Instant::now(), format!("s{si}_trans"), { transpose_full(&h, t, st, nth) });
        t *= st.stride;
        ch = st.out_c;
        for (ri, ru) in st.res.iter().enumerate() {
            let mut r = h.clone();
            pt!(prof, Instant::now(), format!("s{si}_r{ri}_snk"), {
                snake1d_par(&mut r, ch, t, &ru.alpha0, nth)
            });
            r = pt!(prof, Instant::now(), format!("s{si}_r{ri}_dw"), {
                cv(&r, ch, t, &ru.dw_w, &ru.dw_b, ch, 7, RES_DILATION[ri], ch, RES_PAD[ri], nth)
            });
            pt!(prof, Instant::now(), format!("s{si}_r{ri}_snk"), {
                snake1d_par(&mut r, ch, t, &ru.alpha1, nth)
            });
            r = pt!(prof, Instant::now(), format!("s{si}_r{ri}_pw"), {
                pw1(&r, t, &ru.pw_gemm, &ru.pw_bias_pad, ch, nth)
            });
            h = h.iter().zip(r.iter()).map(|(a, b)| a + b).collect();
        }
        let _ = st.in_c;
    }
    pt!(prof, Instant::now(), "final_snake", {
        snake1d_par(&mut h, 96, t, &w.final_alpha, nth)
    });
    h = pt!(prof, Instant::now(), "final_conv", {
        cv(&h, 96, t, &w.final_w, &w.final_b, 1, 7, 1, 1, 3, nth)
    });
    pt!(prof, Instant::now(), "tanh", { h.iter().map(|v| v.tanh()).collect() })
}
