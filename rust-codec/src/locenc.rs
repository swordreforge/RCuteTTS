//! AudioLocEnc + locenc_to_lm_proj (embed_acoustic_latents path).
//!
//! Source: `model/CuteTTS-distill/weights/tts/model.safetensors`,
//! `locenc.*` (22 tensors) + `locenc_to_lm_proj.* (2 tensors), all bf16.
//! Production runs bf16 (model.py:92); Rust converts to f32 at load with
//! [`crate::simd::bf16_to_f32`] (M8) and computes fp32. Measured dtype gap
//! vs production bf16: 3~8e-2 abs on outputs ~8 (see dump script output).
//!
//! Structure: in_proj 64->1024 + special_token [1,1,1,1024], then per patch
//! [CLS, 2 frames] through 2 plain LocalTransformer layers (no AdaLN,
//! GQA 16Q/2KV, RoPE 1e4, seq=3), CLS readout, final norm, 1024->1024 proj.

use crate::dit::{
    linear_rows, rms_norm_row, rope_apply, rope_tables_for, softmax_row, DitLinear, DitMlp,
    FFN, HEAD_DIM, HIDDEN, N_HEADS, N_KV,
};
use crate::gemm::pack_a;
use crate::simd::bf16_to_f32;
use safetensors::SafeTensors;
use std::collections::HashMap;

pub const LOC_PATCH: usize = 2;
pub const LOC_SEQ: usize = LOC_PATCH + 1; // [CLS, 2 frames]
pub const N_LOC_LAYERS: usize = 2;

pub struct LocLayerW {
    pub norm1: Vec<f32>,
    pub norm2: Vec<f32>,
    pub q: DitLinear,
    pub k: DitLinear,
    pub v: DitLinear,
    pub o: DitLinear,
    pub mlp: DitMlp,
}

pub struct LocencW {
    pub in_w: crate::gemm::PackedA,
    pub in_b: Vec<f32>,
    pub special: Vec<f32>, // [1024]
    pub layers: [LocLayerW; N_LOC_LAYERS],
    pub final_norm: Vec<f32>,
    pub proj: DitLinear, // locenc_to_lm_proj 1024->1024
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

fn lin_bf16(
    map: &HashMap<String, (Vec<usize>, Vec<f32>)>,
    prefix: &str,
    o: usize,
    i: usize,
) -> DitLinear {
    let wkey = prefix.to_string() + ".weight";
    let (s, w) = map.get(&wkey).unwrap_or_else(|| panic!("missing {prefix}.weight")).clone();
    assert_eq!(s, vec![o, i], "shape for {prefix}.weight");
    let b = match map.get(&(prefix.to_string() + ".bias")) {
        Some((sb, bv)) => {
            assert_eq!(*sb, vec![o]);
            bv.clone()
        }
        None => vec![0.0f32; o],
    };
    let g = pack_a(&w, o, i);
    let mut bp = vec![0.0f32; g.rows];
    bp[..o].copy_from_slice(&b);
    DitLinear { w: g, b: bp, out_dim: o }
}

pub fn load_locenc_weights(path: &std::path::Path) -> LocencW {
    let bytes = std::fs::read(path).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let mut map: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    for name in st.names() {
        let short = if let Some(r) = name.strip_prefix("locenc.") {
            format!("locenc.{r}")
        } else if let Some(r) = name.strip_prefix("locenc_to_lm_proj.") {
            format!("proj.{r}")
        } else {
            continue;
        };
        let tv = st.tensor(name).unwrap();
        let shape = tv.shape().to_vec();
        let v = f32_of(&tv);
        assert_eq!(v.len(), shape.iter().product::<usize>(), "{name}");
        map.insert(short, (shape, v));
    }
    assert_eq!(map.len(), 24, "locenc+proj tensor count, got {}", map.len());

    let (s_in, w_in) = map["locenc.in_proj.weight"].clone();
    assert_eq!(s_in, vec![HIDDEN, 64]);
    let in_g = pack_a(&w_in, HIDDEN, 64);
    let mut in_bp = vec![0.0f32; in_g.rows];
    in_bp[..HIDDEN].copy_from_slice(&map["locenc.in_proj.bias"].1);
    let (s_sp, v_sp) = map["locenc.special_token"].clone();
    assert_eq!(s_sp, vec![1, 1, 1, HIDDEN]);

    let mut layers = Vec::with_capacity(N_LOC_LAYERS);
    for li in 0..N_LOC_LAYERS {
        let lb = format!("locenc.encoder.layers.{li}");
        let ab = format!("{lb}.self_attn");
        let mb = format!("{lb}.mlp");
        let nw = |n: &str| map[n].1.clone();
        layers.push(LocLayerW {
            norm1: nw(&format!("{lb}.input_layernorm.weight")),
            norm2: nw(&format!("{lb}.post_attention_layernorm.weight")),
            q: lin_bf16(&map, &format!("{ab}.q_proj"), HIDDEN, HIDDEN),
            k: lin_bf16(&map, &format!("{ab}.k_proj"), N_KV * HEAD_DIM, HIDDEN),
            v: lin_bf16(&map, &format!("{ab}.v_proj"), N_KV * HEAD_DIM, HIDDEN),
            o: lin_bf16(&map, &format!("{ab}.o_proj"), HIDDEN, HIDDEN),
            mlp: DitMlp {
                gate: lin_bf16(&map, &format!("{mb}.gate_proj"), FFN, HIDDEN),
                up: lin_bf16(&map, &format!("{mb}.up_proj"), FFN, HIDDEN),
                down: lin_bf16(&map, &format!("{mb}.down_proj"), HIDDEN, FFN),
            },
        });
    }
    LocencW {
        in_w: in_g,
        in_b: in_bp,
        special: v_sp,
        layers: [layers.remove(0), layers.remove(0)],
        final_norm: map["locenc.encoder.norm.weight"].1.clone(),
        proj: lin_bf16(&map, "proj", HIDDEN, HIDDEN),
    }
}

/// One plain LocalTransformerLayer (no AdaLN): h += attn(norm(h)), h += mlp(norm(h)).
fn loc_layer(h: &mut [f32], lw: &LocLayerW, nth: usize, cos: &[f32], sin: &[f32]) {
    const S: usize = LOC_SEQ;
    let mut normed = vec![0.0f32; S * HIDDEN];
    for s in 0..S {
        let n = rms_norm_row(&h[s * HIDDEN..(s + 1) * HIDDEN], &lw.norm1);
        normed[s * HIDDEN..(s + 1) * HIDDEN].copy_from_slice(&n);
    }
    let q = linear_rows(&normed, S, &lw.q, nth);
    let k = linear_rows(&normed, S, &lw.k, nth);
    let v = linear_rows(&normed, S, &lw.v, nth);
    let mut attn_out = vec![0.0f32; S * HIDDEN];
    let mut scores = [0.0f32; S * S];
    let mut vv = [0.0f32; HEAD_DIM];
    for hh in 0..N_HEADS {
        let kh = hh / (N_HEADS / N_KV);
        let mut qr = [[0.0f32; HEAD_DIM]; S];
        let mut kr = [[0.0f32; HEAD_DIM]; S];
        for s in 0..S {
            let mut qv = [0.0f32; HEAD_DIM];
            let mut kv = [0.0f32; HEAD_DIM];
            for d in 0..HEAD_DIM {
                qv[d] = q[s * HIDDEN + hh * HEAD_DIM + d];
                kv[d] = k[s * (N_KV * HEAD_DIM) + kh * HEAD_DIM + d];
            }
            qr[s] = rope_apply(&qv, s, cos, sin).try_into().unwrap();
            kr[s] = rope_apply(&kv, s, cos, sin).try_into().unwrap();
        }
        for i in 0..S {
            for j in 0..S {
                let mut acc = 0.0f32;
                for d in 0..HEAD_DIM {
                    acc += qr[i][d] * kr[j][d];
                }
                scores[i * S + j] = acc * (1.0 / 8.0);
            }
            softmax_row(&mut scores[i * S..(i + 1) * S]);
        }
        for i in 0..S {
            for d in 0..HEAD_DIM {
                vv[d] = 0.0;
            }
            for j in 0..S {
                let p = scores[i * S + j];
                for d in 0..HEAD_DIM {
                    vv[d] += p * v[j * (N_KV * HEAD_DIM) + kh * HEAD_DIM + d];
                }
            }
            for d in 0..HEAD_DIM {
                attn_out[i * HIDDEN + hh * HEAD_DIM + d] = vv[d];
            }
        }
    }
    let proj = linear_rows(&attn_out, S, &lw.o, nth);
    for i in 0..h.len() {
        h[i] += proj[i];
    }
    let mut normed2 = vec![0.0f32; S * HIDDEN];
    for s in 0..S {
        let n = rms_norm_row(&h[s * HIDDEN..(s + 1) * HIDDEN], &lw.norm2);
        normed2[s * HIDDEN..(s + 1) * HIDDEN].copy_from_slice(&n);
    }
    let gate = linear_rows(&normed2, S, &lw.mlp.gate, nth);
    let up = linear_rows(&normed2, S, &lw.mlp.up, nth);
    let mut act = vec![0.0f32; S * FFN];
    for i in 0..act.len() {
        let g = gate[i] / (1.0 + (-gate[i]).exp());
        act[i] = g * up[i];
    }
    let down = linear_rows(&act, S, &lw.mlp.down, nth);
    for i in 0..h.len() {
        h[i] += down[i];
    }
}

/// Full embed_acoustic_latents: x[B,T,2,64] -> [B,T,1024].
pub fn locenc_embed(w: &LocencW, x: &[f32], b: usize, t: usize, nth: usize) -> Vec<f32> {
    assert_eq!(x.len(), b * t * LOC_PATCH * 64);
    // in_proj per frame: [B*T*2, 64] -> [B*T*2, 1024]
    let nf = b * t * LOC_PATCH;
    let hidden = linear_rows_in(x, nf, &w.in_w, &w.in_b, HIDDEN, nth);
    // assemble [B*T, 3, 1024] with CLS first
    let mut seq = vec![0.0f32; b * t * LOC_SEQ * HIDDEN];
    for bt in 0..b * t {
        seq[(bt * LOC_SEQ) * HIDDEN..(bt * LOC_SEQ + 1) * HIDDEN].copy_from_slice(&w.special);
        for p in 0..LOC_PATCH {
            let src = (bt * LOC_PATCH + p) * HIDDEN;
            let dst = (bt * LOC_SEQ + 1 + p) * HIDDEN;
            seq[dst..dst + HIDDEN].copy_from_slice(&hidden[src..src + HIDDEN]);
        }
    }
    let (cos, sin) = rope_tables_for(LOC_SEQ);
    for bt in 0..b * t {
        let h = &mut seq[bt * LOC_SEQ * HIDDEN..(bt + 1) * LOC_SEQ * HIDDEN];
        for lw in &w.layers {
            loc_layer(h, lw, nth, &cos, &sin);
        }
        for s in 0..LOC_SEQ {
            let n = rms_norm_row(&h[s * HIDDEN..(s + 1) * HIDDEN], &w.final_norm);
            h[s * HIDDEN..(s + 1) * HIDDEN].copy_from_slice(&n);
        }
    }
    // CLS readout + proj
    let mut cls = vec![0.0f32; b * t * HIDDEN];
    for bt in 0..b * t {
        cls[bt * HIDDEN..(bt + 1) * HIDDEN]
            .copy_from_slice(&seq[bt * LOC_SEQ * HIDDEN..bt * LOC_SEQ * HIDDEN + HIDDEN]);
    }
    linear_rows(&cls, b * t, &w.proj, nth)
}

/// [`linear_rows`] over a pre-packed weight, chunked to ≤8 rows per
/// micro-kernel block (LocEnc prefill can feed hundreds of rows).
fn linear_rows_in(x: &[f32], s: usize, w: &crate::gemm::PackedA, b: &[f32], o: usize, nth: usize) -> Vec<f32> {
    let i = w.cols;
    assert_eq!(x.len(), s * i);
    const T: usize = 8;
    let mut out = vec![0.0f32; s * o];
    for sb in (0..s).step_by(T) {
        let sn = (sb + T).min(s) - sb;
        let mut xb = vec![0.0f32; i * T];
        for ii in 0..i {
            for ss in 0..sn {
                xb[ii * T + ss] = x[(sb + ss) * i + ii];
            }
        }
        let ln = if (o as u64) * (i as u64) * (sn as u64) >= 8_000_000 { nth } else { 1 };
        let mut c = vec![0.0f32; w.rows * T];
        crate::gemm::sgemm_bias(w, &xb, T, b, &mut c, ln);
        for ss in 0..sn {
            for oo in 0..o {
                out[(sb + ss) * o + oo] = c[oo * T + ss];
            }
        }
    }
    out
}
