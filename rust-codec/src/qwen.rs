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
