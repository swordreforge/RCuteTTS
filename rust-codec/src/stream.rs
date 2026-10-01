//! Streaming (chunked) AudioVAE decode with explicit conv state.
//!
//! Mirrors `AudioStreamingVAEDecoder` (modeling/audio_adapter.py:105-232)
//! WITHOUT monkey-patching: each causal conv keeps its left-history
//! explicitly (fed to [`crate::conv::valid_conv1d` — NOT the causal kernel,
//! which would double-pad), transposes keep 1 past input (context =
//! (k-1)//s = 1 for all four stages; dependency window < 2 input steps).
//! Snake/Tanh/1x1 are stateless. Same taps as offline => bit-identical.

use crate::conv::default_threads;
use crate::conv::valid_conv1d;
use crate::decode::{pw1, transpose_full};
use crate::snake::snake1d_nth as snake_par;
use crate::weights::DecoderW;
use crate::decode::{RES_DILATION, RES_PAD};

/// Left-history buffer: [C, H] row-major, zeros at stream start.
struct Hist {
    c: usize,
    h: usize,
    buf: Vec<f32>,
}

impl Hist {
    fn new(c: usize, h: usize) -> Self {
        Hist { c, h, buf: vec![0.0f32; c * h] }
    }

    fn reset(&mut self) {
        self.buf.fill(0.0);
    }

    /// Returns [C, H+T] = history ++ x; advances history to the last H
    /// columns of the extended buffer.
    fn extend(&mut self, x: &[f32], t: usize) -> Vec<f32> {
        assert_eq!(x.len(), self.c * t);
        // Per-row copies: ext rows are (H+T) wide, history rows H wide;
        // a flat copy misaligns rows (found in M17 review).
        let mut ext = vec![0.0f32; self.c * (self.h + t)];
        for c in 0..self.c {
            ext[c * (self.h + t)..c * (self.h + t) + self.h]
                .copy_from_slice(&self.buf[c * self.h..(c + 1) * self.h]);
            ext[c * (self.h + t) + self.h..(c + 1) * (self.h + t)]
                .copy_from_slice(&x[c * t..(c + 1) * t]);
            let src = &ext[c * (self.h + t) + t..(c + 1) * (self.h + t)];
            self.buf[c * self.h..(c + 1) * self.h].copy_from_slice(src);
        }
        ext
    }
}

pub struct StreamingDecoder<'w> {
    w: &'w DecoderW,
    nth: usize,
    front_dw: Hist,          // 64ch x 6
    trans: [Hist; 4],        // stage in_ch x 1
    res_dw: [[Hist; 3]; 4],  // stage out_ch x 6*d
    final_c: Hist,           // 96ch x 6
}

impl<'w> StreamingDecoder<'w> {
    pub fn new(w: &'w DecoderW, nth: usize) -> Self {
        let ch = [1536usize, 768, 384, 192];
        let res_ch = [768usize, 384, 192, 96];
        let res_dw = std::array::from_fn(|si| {
            std::array::from_fn(|ri| Hist::new(res_ch[si], RES_PAD[ri] * 2))
        });
        StreamingDecoder {
            w,
            nth,
            front_dw: Hist::new(64, 6),
            trans: std::array::from_fn(|si| Hist::new(ch[si], 1)),
            res_dw,
            final_c: Hist::new(96, 6),
        }
    }

    pub fn with_threads(w: &'w DecoderW) -> Self {
        Self::new(w, default_threads())
    }

    pub fn reset(&mut self) {
        self.front_dw.reset();
        for h in self.trans.iter_mut() {
            h.reset();
        }
        for st in self.res_dw.iter_mut() {
            for h in st.iter_mut() {
                h.reset();
            }
        }
        self.final_c.reset();
    }

    /// Decode one latent chunk [64, F] (F=2 per AR patch) -> F*1920 samples.
    pub fn decode_chunk(&mut self, latent: &[f32], frames: usize) -> Vec<f32> {
        assert_eq!(latent.len(), 64 * frames);
        let nth = self.nth;
        let mut h = {
            let ext = self.front_dw.extend(latent, frames);
            valid_conv1d(&ext, 64, 6 + frames, &self.w.front_dw_w, &self.w.front_dw_b, 64, 7, 1, 64, nth)
        };
        h = pw1(&h, frames, &self.w.front_pw_gemm, &self.w.front_pw_bias_pad, 1536, nth);
        let (mut t, mut ch) = (frames, 1536);
        for (si, st) in self.w.stages.iter().enumerate() {
            snake_par(&mut h, ch, t, &st.alpha, nth);
            {
                let ext = self.trans[si].extend(&h, t);
                let full = transpose_full(&ext, 1 + t, st, nth);
                // rows [s .. s+T*s] (drop left transient s samples)
                let s = st.stride;
                let mut out = vec![0.0f32; st.out_c * t * s];
                for o in 0..st.out_c {
                    out[o * t * s..(o + 1) * t * s]
                        .copy_from_slice(&full[o * (1 + t) * s + s..o * (1 + t) * s + s + t * s]);
                }
                h = out;
            }
            t *= st.stride;
            ch = st.out_c;
            for (ri, ru) in st.res.iter().enumerate() {
                let mut r = h.clone();
                snake_par(&mut r, ch, t, &ru.alpha0, nth);
                {
                    let ext = self.res_dw[si][ri].extend(&r, t);
                    r = valid_conv1d(&ext, ch, t + RES_PAD[ri] * 2, &ru.dw_w, &ru.dw_b, ch, 7, RES_DILATION[ri], ch, nth);
                }
                snake_par(&mut r, ch, t, &ru.alpha1, nth);
                r = pw1(&r, t, &ru.pw_gemm, &ru.pw_bias_pad, ch, nth);
                h = h.iter().zip(r.iter()).map(|(a, b)| a + b).collect();
            }
            let _ = st.in_c;
        }
        snake_par(&mut h, 96, t, &self.w.final_alpha, nth);
        let ext = self.final_c.extend(&h, t);
        h = valid_conv1d(&ext, 96, t + 6, &self.w.final_w, &self.w.final_b, 1, 7, 1, 1, nth);
        h.iter().map(|v| v.tanh()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm::pack_a;
    use crate::weights::StageW;

    fn fake_stage(s: usize, o: usize, i: usize) -> StageW {
        // trans_gemm rows [M0_0..M0_{s-1}, M1_0..M1_{s-1}], each O x I
        let mut st = std::collections::hash_map::DefaultHasher::new();
        let mut rng = |n: usize| {
            use std::hash::{Hash, Hasher};
            ((n as u64).wrapping_mul(6364136223846793005) + 0x9e3779b97f4a7c15).hash(&mut st);
            (st.finish() % 2000) as f32 * 0.01 - 10.0
        };
        let mat: Vec<f32> = (0..2 * s * o * i).map(|n| rng(n)).collect();
        let g = pack_a(&mat, 2 * s * o, i);
        let mut bp = vec![0.0f32; g.rows];
        for r in 0..s * o {
            bp[r] = (r % 7) as f32 * 0.1;
        }
        StageW {
            alpha: vec![1.0; i],
            trans_gemm: g,
            trans_bias_pad: bp,
            in_c: i,
            out_c: o,
            stride: s,
            res: std::array::from_fn(|_| crate::weights::ResUnitW {
                alpha0: vec![1.0],
                dw_w: vec![],
                dw_b: vec![],
                alpha1: vec![1.0],
                pw_gemm: pack_a(&[0.0f32; 1], 1, 1),
                pw_bias_pad: vec![0.0],
                dim: 0,
            }),
        }
    }

    #[test]
    fn transpose_slice_matches_offline() {        // slice(transpose_full([0 ++ x])) == transpose_full(x):
        // the streaming transpose claim, all four stage geometries.
        for (s, o, i, t) in [(16usize, 8, 12, 5), (8, 6, 10, 7), (5, 4, 6, 4), (3, 5, 7, 6)] {
            let st = fake_stage(s, o, i);
            let mut hst = std::collections::hash_map::DefaultHasher::new();
            let mut rng = |n: usize| {
                use std::hash::{Hash, Hasher};
                ((n as u64).wrapping_mul(2862933555777941757) + 0x9e3779b97f4a7c15).hash(&mut hst);
                (hst.finish() % 2000) as f32 * 0.01 - 10.0
            };
            let x: Vec<f32> = (0..i * t).map(|n| rng(n)).collect();
            let yref = transpose_full(&x, t, &st, 1);
            assert_eq!(yref.len(), o * t * s);
            // per-row: col 0 = 0, cols 1..T = x
            let mut ext = vec![0.0f32; i * (1 + t)];
            for c in 0..i {
                ext[c * (1 + t) + 1..(c + 1) * (1 + t)].copy_from_slice(&x[c * t..(c + 1) * t]);
            }
            let full = transpose_full(&ext, 1 + t, &st, 1);
            assert_eq!(full.len(), o * (1 + t) * s);
            for oo in 0..o {
                for k in 0..t * s {
                    let a = full[oo * (1 + t) * s + s + k];
                    let b = yref[oo * t * s + k];
                    assert!((a - b).abs() < 1e-4, "s={s} o={oo} k={k}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn stage_bisect_front() {
        // Streaming front (extend+valid+pw1) vs offline front (cv+pw1)
        // on real weights + real latent. Isolates the first stage.
        use crate::decode::{cv, pw1};
        use crate::weights::load_decoder_weights;
        let wpath = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../model/CuteTTS/weights/audio_vae/model.safetensors");
        if !wpath.is_file() {
            eprintln!("skip");
            return;
        }
        let w = load_decoder_weights(&wpath);
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/m2_lat_00.npy");
        let lat: ndarray::Array2<f32> = ndarray_npy::read_npy(p).unwrap();
        let latent = lat.as_slice().unwrap().to_vec(); // [64,2]
        let nth = 1;
        // offline front
        let mut href = cv(&latent, 64, 2, &w.front_dw_w, &w.front_dw_b, 64, 7, 1, 64, 3, nth);
        // streaming dw only (before pw1)
        let mut dec = StreamingDecoder::new(&w, nth);
        let ext = dec.front_dw.extend(&latent, 2);
        let dw_only = crate::conv::valid_conv1d(&ext, 64, 8, &w.front_dw_w, &w.front_dw_b, 64, 7, 1, 64, nth);
        // offline dw only for comparison
        let dw_ref = cv(&latent, 64, 2, &w.front_dw_w, &w.front_dw_b, 64, 7, 1, 64, 3, nth);
        println!("ext row0 = {:.4?}", &ext[..8]);
        println!("w row0 = {:.4?}", &w.front_dw_w[..7]);
        println!("latent row0 = {:.4?}", &latent[..2]);
        println!("b[0] = {:.4}", w.front_dw_b[0]);
        let edw: f32 = dw_only.iter().zip(dw_ref.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        println!("dw-only bisect err={edw:.2e}");
        println!("dw_only[0..4]={:.4?} dw_ref[0..4]={:.4?}", &dw_only[..4], &dw_ref[..4]);
        // hand check o=0: out[0] = b[0] + latent[0]*w[6] (only kk=6 survives zero history)
        let hand = w.front_dw_b[0] + latent[0] * w.front_dw_w[6];
        println!("hand o=0: {hand:.4}");
        href = pw1(&href, 2, &w.front_pw_gemm, &w.front_pw_bias_pad, 1536, nth);
        // streaming front
        let mut dec = StreamingDecoder::new(&w, nth);
        let ext = dec.front_dw.extend(&latent, 2);
        let got = crate::conv::valid_conv1d(&ext, 64, 8, &w.front_dw_w, &w.front_dw_b, 64, 7, 1, 64, nth);
        let got = pw1(&got, 2, &w.front_pw_gemm, &w.front_pw_bias_pad, 1536, nth);
        let e: f32 = href.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        println!("front bisect err={e:.2e}");
        assert!(e < 1e-6, "front {e:.2e}");
    }
}
