//! Qwen3-7L backbone weights (distill): 79 tensors / 126.9M, all bf16.
//!
//! Source: `model/CuteTTS-distill/weights/tts/model.safetensors`, `qwen_backbone.*`.
//! Fresh 7-layer Qwen3 (model.py builds with num_hidden_layers=7, not truncated):
//! hidden 1024, heads 16, kv 8, head_dim 128, SwiGLU intermediate 3072,
//! QK-Norm (q_norm/k_norm [128] per head), RoPE theta 1e6, no sliding window,
//! no lm_head (AutoModel), embed resized to 16385.
//! Production runs bf16; Rust converts to f32 at load ([`crate::simd::bf16_to_f32`]).
//! Measured dtype gap vs production bf16: 2.8e-1 abs on outputs ~19 (decode).

use crate::dit::DitLinear;
use crate::gemm::pack_a;
use crate::simd::bf16_to_f32;
use safetensors::SafeTensors;
use std::collections::HashMap;

pub const Q_HIDDEN: usize = 1024;
pub const Q_LAYERS: usize = 7;
pub const Q_HEADS: usize = 16;
pub const Q_KV: usize = 8;
pub const Q_HEAD_DIM: usize = 128;
pub const Q_FFN: usize = 3072;
pub const Q_VOCAB: usize = 16385;
pub const Q_ROPE_THETA: f32 = 1_000_000.0;

pub struct QwenLayerW {
    pub norm1: Vec<f32>,
    pub norm2: Vec<f32>,
    pub q: DitLinear,
    pub k: DitLinear,
    pub v: DitLinear,
    pub o: DitLinear,
    pub qn: Vec<f32>, // QK-Norm weights [128], shared across heads
    pub kn: Vec<f32>,
    pub gate: DitLinear,
    pub up: DitLinear,
    pub down: DitLinear,
}

pub struct QwenW {
    pub embed: Vec<f32>, // [16385, 1024] row-major f32
    pub layers: [QwenLayerW; Q_LAYERS],
    pub final_norm: Vec<f32>,
}

fn f32_of(tv: &safetensors::tensor::TensorView<'_>) -> Vec<f32> {
    match tv.dtype() {
        safetensors::Dtype::F32 => tv
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        safetensors::Dtype::BF16 => {
            let src: Vec<u16> = tv
                .data()
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let mut dst = vec![0.0f32; src.len()];
            bf16_to_f32(&mut dst, &src);
            dst
        }
        d => panic!("unexpected dtype {d:?}"),
    }
}

fn qlin(map: &HashMap<String, (Vec<usize>, Vec<f32>)>, name: &str, o: usize, i: usize) -> DitLinear {
    let (s, w) = map.get(name).unwrap_or_else(|| panic!("missing {name}")).clone();
    assert_eq!(s, vec![o, i], "shape for {name}");
    let g = pack_a(&w, o, i);
    let rows = g.rows;
    DitLinear { w: g, b: vec![0.0f32; rows], out_dim: o }
}

pub fn load_qwen_weights(path: &std::path::Path) -> QwenW {
    let bytes = std::fs::read(path).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let mut map: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    for name in st.names() {
        let Some(r) = name.strip_prefix("qwen_backbone.") else { continue };
        let tv = st.tensor(name).unwrap();
        let shape = tv.shape().to_vec();
        let v = f32_of(&tv);
        assert_eq!(v.len(), shape.iter().product::<usize>(), "{name}");
        map.insert(r.to_string(), (shape, v));
    }
    assert_eq!(map.len(), 79, "qwen tensor count, got {}", map.len());

    let (se, ve) = map["embed_tokens.weight"].clone();
    assert_eq!(se, vec![Q_VOCAB, Q_HIDDEN]);

    let mut layers = Vec::with_capacity(Q_LAYERS);
    for li in 0..Q_LAYERS {
        let lb = format!("layers.{li}");
        let ab = format!("{lb}.self_attn");
        let nw = |n: String| map[&n].1.clone();
        layers.push(QwenLayerW {
            norm1: nw(format!("{lb}.input_layernorm.weight")),
            norm2: nw(format!("{lb}.post_attention_layernorm.weight")),
            q: qlin(&map, &format!("{ab}.q_proj.weight"), Q_HEADS * Q_HEAD_DIM, Q_HIDDEN),
            k: qlin(&map, &format!("{ab}.k_proj.weight"), Q_KV * Q_HEAD_DIM, Q_HIDDEN),
            v: qlin(&map, &format!("{ab}.v_proj.weight"), Q_KV * Q_HEAD_DIM, Q_HIDDEN),
            o: qlin(&map, &format!("{ab}.o_proj.weight"), Q_HIDDEN, Q_HEADS * Q_HEAD_DIM),
            qn: nw(format!("{ab}.q_norm.weight")),
            kn: nw(format!("{ab}.k_norm.weight")),
            gate: qlin(&map, &format!("{lb}.mlp.gate_proj.weight"), Q_FFN, Q_HIDDEN),
            up: qlin(&map, &format!("{lb}.mlp.up_proj.weight"), Q_FFN, Q_HIDDEN),
            down: qlin(&map, &format!("{lb}.mlp.down_proj.weight"), Q_HIDDEN, Q_FFN),
        });
    }
    let mut out = Vec::with_capacity(Q_LAYERS);
    for l in layers.drain(..) {
        out.push(l);
    }
    QwenW {
        embed: ve,
        layers: [out.remove(0), out.remove(0), out.remove(0), out.remove(0), out.remove(0), out.remove(0), out.remove(0)],
        final_norm: map["norm.weight"].1.clone(),
    }
}

// ============================================================
// Forward: prefill (GEMM path) + single-token decode (GEMV path).
// Mirrors transformers Qwen3Model with sdpa, mask=None:
// causal prefill over [T,H], decode attends cached K/V + new row.
// No logits (AutoModel, no lm_head); stop comes from stop_predictor.
// ============================================================

use crate::conv::default_threads;
use crate::dit::{linear_rows, rms_norm_row, softmax_row};
use crate::simd::{resolve_saxpy, run_gemv8, SaxpyKind};

const Q_ATTN_SCALE: f32 = 1.0 / 11.3137085; // 1/sqrt(128)
const Q_KV_DIM: usize = Q_KV * Q_HEAD_DIM; // 1024
const Q_Q_DIM: usize = Q_HEADS * Q_HEAD_DIM; // 2048

/// Per-layer K/V cache: flat rows of [Q_KV_DIM], post-RoPE K.
/// (f32 storage; production keeps bf16 — parity gap recorded in M12.)
pub struct QwenCache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub len: usize,
}

impl QwenCache {
    pub fn empty() -> Vec<QwenCache> {
        (0..Q_LAYERS).map(|_| QwenCache { k: vec![], v: vec![], len: 0 }).collect()
    }
}

/// RoPE inv_freq for head_dim 128, theta 1e6 (computed once per model).
pub struct QwenRope {
    pub inv: [f32; 64],
}

impl QwenRope {
    pub fn new() -> Self {
        let mut inv = [0.0f32; 64];
        for i in 0..64 {
            inv[i] = Q_ROPE_THETA.powf(-((2 * i) as f32) / Q_HEAD_DIM as f32);
        }
        QwenRope { inv }
    }

    /// NeoX rotate_half apply at absolute position pos (v len 128).
    pub fn apply(&self, v: &[f32], pos: usize, out: &mut [f32]) {
        debug_assert_eq!(v.len(), Q_HEAD_DIM);
        for i in 0..64 {
            let (s, c) = (pos as f32 * self.inv[i]).sin_cos();
            out[i] = v[i] * c - v[64 + i] * s;
            out[64 + i] = v[i] * s + v[64 + i] * c;
        }
    }
}

/// GEMV y[O] = W[O,I] x[I] + b over packed panels (decode path).
fn linear_gemv(l: &DitLinear, x: &[f32], kind: SaxpyKind) -> Vec<f32> {
    let (o, k) = (l.out_dim, l.w.cols);
    assert_eq!(x.len(), k);
    let panels = l.w.rows / 8;
    // Pool-dispatch big GEMVs (>= 1M MACs); small ones stay serial.
    // Bitwise identical either way (disjoint panels).
    let nth = crate::conv::default_threads();
    let ops = o as u64 * k as u64;
    if nth <= 1 || ops < 1_000_000 || panels <= 1 {
        return linear_gemv_serial(l, x, kind);
    }
    let per = (panels + nth - 1) / nth;
    let mut y = vec![0.0f32; panels * 8];
    let jobs: Vec<Box<dyn FnOnce() + Send + '_>> = y
        .chunks_mut(per * 8)
        .enumerate()
        .map(|(pi, chunk)| {
            let p0 = pi * per;
            let p1 = (p0 + per).min(panels);
            Box::new(move || {
                for p in p0..p1 {
                    let mut bb = [0.0f32; 8];
                    bb.copy_from_slice(&l.b[p * 8..(p + 1) * 8]);
                    run_gemv8(kind, &l.w.data[p * k * 8..(p + 1) * k * 8], x, k, &bb, &mut chunk[(p - p0) * 8..(p - p0 + 1) * 8]);
                }
            }) as Box<dyn FnOnce() + Send + '_>
        })
        .collect();
    crate::pool::scope(jobs);
    y.truncate(o);
    y
}

fn linear_gemv_serial(l: &DitLinear, x: &[f32], kind: SaxpyKind) -> Vec<f32> {
    let (o, k) = (l.out_dim, l.w.cols);
    let panels = l.w.rows / 8;
    let mut y = vec![0.0f32; panels * 8];
    for p in 0..panels {
        let mut bb = [0.0f32; 8];
        bb.copy_from_slice(&l.b[p * 8..(p + 1) * 8]);
        run_gemv8(kind, &l.w.data[p * k * 8..(p + 1) * k * 8], x, k, &bb, &mut y[p * 8..(p + 1) * 8]);
    }
    y.truncate(o);
    y
}

/// QK-Norm: per-head RMSNorm with shared [128] weight.
fn qk_norm(x: &[f32], w: &[f32]) -> Vec<f32> {
    rms_norm_row(x, w)
}

#[allow(clippy::too_many_arguments)]
fn qwen_layer_prefill(
    lw: &QwenLayerW,
    h: &[f32],
    t: usize,
    pos0: usize,
    rope: &QwenRope,
    cache: &mut QwenCache,
    nth: usize,
) -> Vec<f32> {
    // norm + q/k/v
    let mut normed = vec![0.0f32; t * Q_HIDDEN];
    for s in 0..t {
        let n = rms_norm_row(&h[s * Q_HIDDEN..(s + 1) * Q_HIDDEN], &lw.norm1);
        normed[s * Q_HIDDEN..(s + 1) * Q_HIDDEN].copy_from_slice(&n);
    }
    let q = linear_rows(&normed, t, &lw.q, nth);
    let k = linear_rows(&normed, t, &lw.k, nth);
    let v = linear_rows(&normed, t, &lw.v, nth);
    // qk-norm + rope per head per token; append k/v to cache
    let mut qr = vec![0.0f32; t * Q_Q_DIM];
    let mut kr = vec![0.0f32; t * Q_KV_DIM];
    let mut tmp = [0.0f32; Q_HEAD_DIM];
    for s in 0..t {
        for hh in 0..Q_HEADS {
            let n = qk_norm(&q[s * Q_Q_DIM + hh * Q_HEAD_DIM..s * Q_Q_DIM + (hh + 1) * Q_HEAD_DIM], &lw.qn);
            rope.apply(&n, pos0 + s, &mut tmp);
            qr[s * Q_Q_DIM + hh * Q_HEAD_DIM..s * Q_Q_DIM + (hh + 1) * Q_HEAD_DIM].copy_from_slice(&tmp);
        }
        for kh in 0..Q_KV {
            let n = qk_norm(&k[s * Q_KV_DIM + kh * Q_HEAD_DIM..s * Q_KV_DIM + (kh + 1) * Q_HEAD_DIM], &lw.kn);
            rope.apply(&n, pos0 + s, &mut tmp);
            kr[s * Q_KV_DIM + kh * Q_HEAD_DIM..s * Q_KV_DIM + (kh + 1) * Q_HEAD_DIM].copy_from_slice(&tmp);
        }
        cache.k.push(kr[s * Q_KV_DIM..(s + 1) * Q_KV_DIM].to_vec());
        cache.v.push(v[s * Q_KV_DIM..(s + 1) * Q_KV_DIM].to_vec());
    }
    cache.len += t;
    // causal attention (per-token width is 16 heads x 128 = Q_Q_DIM)
    let mut attn_out = vec![0.0f32; t * Q_Q_DIM];
    let mut scores = vec![0.0f32; t];
    for hh in 0..Q_HEADS {
        let kh = hh / (Q_HEADS / Q_KV);
        for i in 0..t {
            for j in 0..=i {
                let mut acc = 0.0f32;
                for d in 0..Q_HEAD_DIM {
                    acc += qr[i * Q_Q_DIM + hh * Q_HEAD_DIM + d] * kr[j * Q_KV_DIM + kh * Q_HEAD_DIM + d];
                }
                scores[j] = acc * Q_ATTN_SCALE;
            }
            for j in i + 1..t {
                scores[j] = f32::NEG_INFINITY;
            }
            softmax_row(&mut scores);
            for d in 0..Q_HEAD_DIM {
                let mut acc = 0.0f32;
                for j in 0..=i {
                    acc += scores[j] * v[j * Q_KV_DIM + kh * Q_HEAD_DIM + d];
                }
                attn_out[i * Q_Q_DIM + hh * Q_HEAD_DIM + d] = acc;
            }
        }
    }
    let proj = linear_rows(&attn_out, t, &lw.o, nth);
    let mut h2 = vec![0.0f32; t * Q_HIDDEN];
    for i in 0..h2.len() {
        h2[i] = h[i] + proj[i];
    }
    let mut normed2 = vec![0.0f32; t * Q_HIDDEN];
    for s in 0..t {
        let n = rms_norm_row(&h2[s * Q_HIDDEN..(s + 1) * Q_HIDDEN], &lw.norm2);
        normed2[s * Q_HIDDEN..(s + 1) * Q_HIDDEN].copy_from_slice(&n);
    }
    let gate = linear_rows(&normed2, t, &lw.gate, nth);
    let up = linear_rows(&normed2, t, &lw.up, nth);
    let act: Vec<f32> = gate.iter().zip(up.iter()).map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u).collect();
    let down = linear_rows(&act, t, &lw.down, nth);
    for i in 0..h2.len() {
        h2[i] += down[i];
    }
    h2
}

fn qwen_layer_decode(
    lw: &QwenLayerW,
    h: &[f32],
    pos: usize,
    rope: &QwenRope,
    cache: &mut QwenCache,
    kind: SaxpyKind,
) -> Vec<f32> {
    assert_eq!(h.len(), Q_HIDDEN);
    let normed = rms_norm_row(h, &lw.norm1);
    let q = linear_gemv(&lw.q, &normed, kind);
    let k = linear_gemv(&lw.k, &normed, kind);
    let v = linear_gemv(&lw.v, &normed, kind);
    let mut qr = vec![0.0f32; Q_Q_DIM];
    let mut kr = vec![0.0f32; Q_KV_DIM];
    let mut tmp = [0.0f32; Q_HEAD_DIM];
    for hh in 0..Q_HEADS {
        let n = qk_norm(&q[hh * Q_HEAD_DIM..(hh + 1) * Q_HEAD_DIM], &lw.qn);
        rope.apply(&n, pos, &mut tmp);
        qr[hh * Q_HEAD_DIM..(hh + 1) * Q_HEAD_DIM].copy_from_slice(&tmp);
    }
    for kh in 0..Q_KV {
        let n = qk_norm(&k[kh * Q_HEAD_DIM..(kh + 1) * Q_HEAD_DIM], &lw.kn);
        rope.apply(&n, pos, &mut tmp);
        kr[kh * Q_HEAD_DIM..(kh + 1) * Q_HEAD_DIM].copy_from_slice(&tmp);
    }
    cache.k.push(kr.clone());
    cache.v.push(v.clone());
    cache.len += 1;
    let n = cache.len;
    let mut attn_out = vec![0.0f32; Q_Q_DIM];
    let mut scores = vec![0.0f32; n];
    for hh in 0..Q_HEADS {
        let kh = hh / (Q_HEADS / Q_KV);
        for j in 0..n {
            let kj = &cache.k[j];
            let mut acc = 0.0f32;
            for d in 0..Q_HEAD_DIM {
                acc += qr[hh * Q_HEAD_DIM + d] * kj[kh * Q_HEAD_DIM + d];
            }
            scores[j] = acc * Q_ATTN_SCALE;
        }
        softmax_row(&mut scores);
        for d in 0..Q_HEAD_DIM {
            let mut acc = 0.0f32;
            for j in 0..n {
                acc += scores[j] * cache.v[j][kh * Q_HEAD_DIM + d];
            }
            attn_out[hh * Q_HEAD_DIM + d] = acc;
        }
    }
    let proj = linear_gemv(&lw.o, &attn_out, kind);    let mut h2 = vec![0.0f32; Q_HIDDEN];
    for i in 0..Q_HIDDEN {
        h2[i] = h[i] + proj[i];
    }
    let normed2 = rms_norm_row(&h2, &lw.norm2);
    let gate = linear_gemv(&lw.gate, &normed2, kind);
    let up = linear_gemv(&lw.up, &normed2, kind);
    let act: Vec<f32> = gate.iter().zip(up.iter()).map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u).collect();
    let down = linear_gemv(&lw.down, &act, kind);
    for i in 0..Q_HIDDEN {
        h2[i] += down[i];
    }
    h2
}

/// Full prefill: x[T,1024] at absolute positions pos0.. -> [T,1024], cache filled.
pub fn prefill(w: &QwenW, x: &[f32], t: usize, pos0: usize, cache: &mut Vec<QwenCache>, nth: usize) -> Vec<f32> {
    assert_eq!(x.len(), t * Q_HIDDEN);
    let rope = QwenRope::new();
    let mut h = x.to_vec();
    for (lw, c) in w.layers.iter().zip(cache.iter_mut()) {
        h = qwen_layer_prefill(lw, &h, t, pos0, &rope, c, nth);
    }
    for s in 0..t {
        let n = rms_norm_row(&h[s * Q_HIDDEN..(s + 1) * Q_HIDDEN], &w.final_norm);
        h[s * Q_HIDDEN..(s + 1) * Q_HIDDEN].copy_from_slice(&n);
    }
    h
}

/// Single-token decode: x[1024] at absolute position pos -> [1024].
pub fn decode_step(w: &QwenW, x: &[f32], pos: usize, cache: &mut Vec<QwenCache>) -> Vec<f32> {
    // Serial by design: decode GEMVs are small (<=3.1M MACs); panel
    // threading would be pure spawn overhead. Threading happens one
    // level up (across utterances), not here.
    assert_eq!(x.len(), Q_HIDDEN);
    let kind = resolve_saxpy();
    let rope = QwenRope::new();
    let mut h = x.to_vec();
    for (lw, c) in w.layers.iter().zip(cache.iter_mut()) {
        h = qwen_layer_decode(lw, &h, pos, &rope, c, kind);
    }
    rms_norm_row(&h, &w.final_norm)
}

/// Embedding lookup: ids -> [N, 1024] rows (bit-exact gather).
pub fn embed_lookup(w: &QwenW, ids: &[i64]) -> Vec<f32> {
    let mut out = vec![0.0f32; ids.len() * Q_HIDDEN];
    for (s, &id) in ids.iter().enumerate() {
        assert!((id as usize) < Q_VOCAB, "token id {id} out of range");
        out[s * Q_HIDDEN..(s + 1) * Q_HIDDEN]
            .copy_from_slice(&w.embed[id as usize * Q_HIDDEN..(id as usize + 1) * Q_HIDDEN]);
    }
    out
}

/// [`decode_step`] with default thread count (prefill threads via nth).
pub fn prefill_default(w: &QwenW, x: &[f32], t: usize, pos0: usize, cache: &mut Vec<QwenCache>) -> Vec<f32> {
    prefill(w, x, t, pos0, cache, default_threads())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_apply_vs_torch() {        // torch apply_rotary_pos_emb on random q, head0 pos1.
        let v = td_vec("dbg_rope_v");
        let refr = td_vec("dbg_rope_out");
        assert_eq!((v.len(), refr.len()), (128, 128));
        let rope = QwenRope::new();
        let mut out = [0.0f32; 128];
        rope.apply(&v, 1, &mut out);
        let e: f32 = out.iter().zip(refr.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        println!("rope apply pos1: err={e:.2e}");
        assert!(e < 1e-5, "rope err={e}");
    }

    #[test]
    fn layer6_causal_clean_checks() {
        // tok0 of the dbg6 fixtures came from an UNMASKED manual attention
        // call (eager mask=None is bidirectional) — only tok1 (last token)
        // is causal-consistent. Assert exactly those quantities:
        // scores, mix, v, post-proj tok1, mlp-tail tok1.
        let wpath = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../model/CuteTTS-distill/weights/tts/model.safetensors");
        if !wpath.is_file() {
            eprintln!("skip: weights missing");
            return;
        }
        let w = load_qwen_weights(&wpath);
        let lw = &w.layers[6];
        let rope = QwenRope::new();
        let h6 = td("qwen_h6");
        let t = 2;
        let inp: Vec<f32> = h6[..t * 1024].to_vec();
        let mut normed = vec![0.0f32; t * 1024];
        for s in 0..t {
            let n = crate::dit::rms_norm_row(&inp[s * 1024..(s + 1) * 1024], &lw.norm1);
            normed[s * 1024..(s + 1) * 1024].copy_from_slice(&n);
        }
        let q = crate::dit::linear_rows(&normed, t, &lw.q, 1);
        let k = crate::dit::linear_rows(&normed, t, &lw.k, 1);
        let v = crate::dit::linear_rows(&normed, t, &lw.v, 1);
        let mut qr = vec![0.0f32; t * 2048];
        let mut kr = vec![0.0f32; t * 1024];
        let mut tmp = [0.0f32; 128];
        for s in 0..t {
            for hh in 0..16 {
                let n = crate::dit::rms_norm_row(&q[s * 2048 + hh * 128..s * 2048 + (hh + 1) * 128], &lw.qn);
                rope.apply(&n, s, &mut tmp);
                qr[s * 2048 + hh * 128..s * 2048 + (hh + 1) * 128].copy_from_slice(&tmp);
            }
            for kh in 0..8 {
                let n = crate::dit::rms_norm_row(&k[s * 1024 + kh * 128..s * 1024 + (kh + 1) * 128], &lw.kn);
                rope.apply(&n, s, &mut tmp);
                kr[s * 1024 + kh * 128..s * 1024 + (kh + 1) * 128].copy_from_slice(&tmp);
            }
        }
        let rd = |n: &str| {
            let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("testdata/{n}.npy"));
            let a: ndarray::Array2<f32> = ndarray_npy::read_npy(p).unwrap();
            a.into_raw_vec_and_offset().0
        };
        // v rows (both tokens causal-clean: v has no mask dependence).
        // NOTE dbg6_v.npy is flat 1D [8*2*128] kh-major.
        let pvv = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/dbg6_v.npy");
        let va: ndarray::Array1<f32> = ndarray_npy::read_npy(pvv).unwrap();
        let vv = va.to_vec();
        let mut ev = 0.0f32;
        for kh in 0..8 {
            for s in 0..t {
                for d in 0..128 {
                    ev = ev.max((v[s * 1024 + kh * 128 + d] - vv[(kh * 2 + s) * 128 + d]).abs());
                }
            }
        }
        println!("layer6 v-rows: err={ev:.2e}");
        assert!(ev < 1e-4, "v err={ev}");
        // tok1 scores / mix / post-proj / mlp-tail (causal-clean)
        let mut attn_out = vec![0.0f32; t * 2048];
        let mut scores = vec![0.0f32; t];
        let mut my_scores = vec![0.0f32; 16 * 2];
        for hh in 0..16 {
            let kh = hh / 2;
            for i in 0..t {
                for j in 0..=i {
                    let mut acc = 0.0f32;
                    for d in 0..128 {
                        acc += qr[i * 2048 + hh * 128 + d] * kr[j * 1024 + kh * 128 + d];
                    }
                    scores[j] = acc * Q_ATTN_SCALE;
                }
                if i == 1 {
                    my_scores[hh * 2] = scores[0];
                    my_scores[hh * 2 + 1] = scores[1];
                }
                for j in i + 1..t {
                    scores[j] = f32::NEG_INFINITY;
                }
                crate::dit::softmax_row(&mut scores);
                for d in 0..128 {
                    let mut acc = 0.0f32;
                    for j in 0..=i {
                        acc += scores[j] * v[j * 1024 + kh * 128 + d];
                    }
                    attn_out[i * 2048 + hh * 128 + d] = acc;
                }
            }
        }
        let es = max_err(&my_scores, &rd("dbg6_scores"));
        println!("layer6 tok1 scores: err={es:.2e}");
        assert!(es < 1e-3, "scores err={es}");
        let em = max_err(&attn_out[2048..], &rd("dbg6_mix")[2048..]);
        println!("layer6 tok1 mix: err={em:.2e}");
        assert!(em < 1e-3, "mix err={em}");
        let proj = crate::dit::linear_rows(&attn_out, t, &lw.o, 1);
        let e1 = max_err(&proj[1024..], &rd("dbg6_attn")[1024..]);
        println!("layer6 tok1 postproj: err={e1:.2e}");
        assert!(e1 < 1e-3, "postproj err={e1}");
        let h6t1: Vec<f32> = h6[1024..2048].to_vec();
        let mut h2t1 = vec![0.0f32; 1024];
        for d in 0..1024 {
            h2t1[d] = h6t1[d] + proj[1024 + d];
        }
        let n2 = crate::dit::rms_norm_row(&h2t1, &lw.norm2);
        let g = crate::dit::linear_rows(&n2, 1, &lw.gate, 1);
        let u = crate::dit::linear_rows(&n2, 1, &lw.up, 1);
        let act: Vec<f32> = g.iter().zip(u.iter()).map(|(&gg, &uu)| (gg / (1.0 + (-gg).exp())) * uu).collect();
        let dn = crate::dit::linear_rows(&act, 1, &lw.down, 1);
        let mall: Vec<f32> = rd("dbg6_mlp");
        let emlp = max_err(&dn, &mall[1024..2048]);
        println!("layer6 tok1 mlp-tail: err={emlp:.2e}");
        assert!(emlp < 1e-3, "mlp err={emlp}");
    }
    fn td(name: &str) -> Vec<f32> {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("testdata/{name}.npy"));
        let a: ndarray::Array2<f32> = ndarray_npy::read_npy(p).unwrap();
        a.into_raw_vec_and_offset().0
    }

    fn td_vec(name: &str) -> Vec<f32> {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("testdata/{name}.npy"));
        let a: ndarray::Array1<f32> = ndarray_npy::read_npy(p).unwrap();
        a.to_vec()
    }

    fn max_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    #[test]
    fn qk_rope_stages_vs_torch() {
        // Stage-by-stage for token 1: raw q-proj -> qn -> rope, k rope.
        // NOTE: this fn lacked #[test] and never ran; enabled for verification.
        // T=1 passed, so norm/mlp/o_proj are trusted; this isolates the Q side.
        for li in [0usize, 6usize] {
            println!("== layer {li} stages ==");
            stage_one_layer(li);
        }
    }

    fn stage_one_layer(_li: usize) {
        // NOTE: `_li` is intentionally unused — the dbg_* fixtures below are
        // layer-0 data, so both loop iterations check layer 0 (diagnostic
        // probe, prints only, no asserts; cf. layer6_causal_clean_checks
        // for the layer-6 gate).
        let wpath = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../model/CuteTTS-distill/weights/tts/model.safetensors");
        if !wpath.is_file() {
            eprintln!("skip: weights missing");
            return;
        }
        let w = load_qwen_weights(&wpath);
        let lw = &w.layers[0];
        let rope = QwenRope::new();
        let h0 = td("qwen_h0");
        let tok1: Vec<f32> = h0[1024..2048].to_vec();
        let normed = crate::dit::rms_norm_row(&tok1, &lw.norm1);
        let q = crate::dit::linear_rows(&normed, 1, &lw.q, 1);
        for (tag, got, want) in [
            ("raw-q", q.clone(), td_vec("dbg_q")),
            ("qn", {
                let mut v = vec![0.0f32; 2048];
                for hh in 0..16 {
                    let n = crate::dit::rms_norm_row(&q[hh * 128..(hh + 1) * 128], &lw.qn);
                    v[hh * 128..(hh + 1) * 128].copy_from_slice(&n);
                }
                v
            }, td_vec("dbg_qn")),
            ("qrope", {
                let mut v = vec![0.0f32; 2048];
                let mut tmp = [0.0f32; 128];
                for hh in 0..16 {
                    let n = crate::dit::rms_norm_row(&q[hh * 128..(hh + 1) * 128], &lw.qn);
                    rope.apply(&n, 1, &mut tmp);
                    v[hh * 128..(hh + 1) * 128].copy_from_slice(&tmp);
                }
                v
            }, td_vec("dbg_qrope")),
            ("krope", {
                let k = crate::dit::linear_rows(&normed, 1, &lw.k, 1);
                let mut v = vec![0.0f32; 1024];
                let mut tmp = [0.0f32; 128];
                for kh in 0..8 {
                    let n = crate::dit::rms_norm_row(&k[kh * 128..(kh + 1) * 128], &lw.kn);
                    rope.apply(&n, 1, &mut tmp);
                    v[kh * 128..(kh + 1) * 128].copy_from_slice(&tmp);
                }
                v
            }, td_vec("dbg_krope")),
        ] {
            let e = max_err(&got, &want);
            println!("{tag}: err={e:.2e}");
        }
    }
}
