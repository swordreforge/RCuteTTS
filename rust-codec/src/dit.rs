//! AudioDiTHead weights (distill): 61 tensors / 73.6M fp32, plain Linears.
//!
//! Source: `model/CuteTTS-distill/weights/tts/model.safetensors`, `head.*`.
//! Unlike the VAE there is NO weight_norm — `nn.Linear` weights are dense
//! `[O, I]` row-major, packed directly with [`crate::gemm::pack_a`].
//! `mu_proj` is `nn.Identity` (cond_dim == hidden_dim), no tensors.
//!
//! Per-layer params: attn q/o 1024² + k/v 128×1024 (GQA 16Q/2KV, head_dim 64,
//! RoPE θ=1e4, bidirectional, seq=5) + SwiGLU MLP 3×(1024×4096) + 2×RMSNorm
//! + speaker AdaLN-Zero 6144×256 (zero-init, bias=False).

use crate::gemm::{pack_a, PackedA};
use safetensors::SafeTensors;
use std::collections::HashMap;

pub const HIDDEN: usize = 1024;
pub const FFN: usize = 4096;
pub const N_HEADS: usize = 16;
pub const N_KV: usize = 2;
pub const HEAD_DIM: usize = 64;
pub const LATENT: usize = 64;
pub const PATCH: usize = 2;
pub const SEQ: usize = 1 + PATCH + PATCH; // [mu+t+dt, cond(2), x(2)] = 5
pub const SPK: usize = 256;
pub const N_LAYERS: usize = 4;

pub struct DitLinear {
    pub w: PackedA,
    /// Padded bias (zeros for bias=False linears).
    pub b: Vec<f32>,
    pub out_dim: usize,
}

pub struct DitMlp {
    pub gate: DitLinear,
    pub up: DitLinear,
    pub down: DitLinear,
}

pub struct DitLayerW {
    pub norm1: Vec<f32>,
    pub norm2: Vec<f32>,
    pub q: DitLinear,
    pub k: DitLinear,
    pub v: DitLinear,
    pub o: DitLinear,
    pub mlp: DitMlp,
    /// speaker AdaLN-Zero: [6144, 256] -> shift/scale/gate x attn/mlp.
    pub adaln: DitLinear,
}

pub struct DitW {
    pub in_proj: DitLinear,
    pub cond_proj: DitLinear,
    pub out_proj: DitLinear,
    pub time_mlp1: DitLinear,
    pub time_mlp2: DitLinear,
    pub delta_mlp1: DitLinear,
    pub delta_mlp2: DitLinear,
    pub step_mlp1: DitLinear,
    pub step_mlp2: DitLinear,
    /// Linear(1,1024,bias=False) -> SiLU -> Linear(1024,1024,bias=False).
    pub cfg_emb0: DitLinear,
    pub cfg_emb2: DitLinear,
    pub layers: [DitLayerW; N_LAYERS],
    pub final_norm: Vec<f32>,
}

fn dense(map: &HashMap<String, (Vec<usize>, Vec<f32>)>, name: &str) -> (Vec<usize>, Vec<f32>) {
    map.get(name).unwrap_or_else(|| panic!("missing tensor {name}")).clone()
}

/// Pack a dense `[O, I]` Linear (+ optional `[O]` bias) for [`crate::gemm`].
fn lin(map: &HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str, o: usize, i: usize) -> DitLinear {
    let (s, w) = dense(map, &format!("{prefix}.weight"));
    assert_eq!(s, vec![o, i], "shape for {prefix}.weight");
    let b = match map.get(&format!("{prefix}.bias")) {
        Some((sb, bv)) => {
            assert_eq!(*sb, vec![o], "shape for {prefix}.bias");
            bv.clone()
        }
        None => vec![0.0f32; o],
    };
    let g = pack_a(&w, o, i);
    let mut bp = vec![0.0f32; g.rows];
    bp[..o].copy_from_slice(&b);
    DitLinear { w: g, b: bp, out_dim: o }
}

fn norm_w(map: &HashMap<String, (Vec<usize>, Vec<f32>)>, name: &str, dim: usize) -> Vec<f32> {
    let (s, v) = dense(map, name);
    assert_eq!(s, vec![dim], "shape for {name}");
    v
}

pub fn load_dit_weights(path: &std::path::Path) -> DitW {
    let bytes = std::fs::read(path).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let mut map: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    for name in st.names() {
        if !name.starts_with("head.") {
            continue;
        }
        let tv = st.tensor(name).unwrap();
        assert_eq!(tv.dtype(), safetensors::Dtype::F32, "{name}");
        let shape = tv.shape().to_vec();
        let v: Vec<f32> = tv
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(v.len(), shape.iter().product::<usize>(), "{name}");
        map.insert(name["head.".len()..].to_string(), (shape, v));
    }
    assert_eq!(map.len(), 61, "distill head tensor count, got {}", map.len());

    let mut layers = Vec::with_capacity(N_LAYERS);
    for li in 0..N_LAYERS {
        let lb = format!("decoder.layers.{li}");
        let ab = format!("{lb}.self_attn");
        let mb = format!("{lb}.mlp");
        layers.push(DitLayerW {
            norm1: norm_w(&map, &format!("{lb}.input_layernorm.weight"), HIDDEN),
            norm2: norm_w(&map, &format!("{lb}.post_attention_layernorm.weight"), HIDDEN),
            q: lin(&map, &format!("{ab}.q_proj"), HIDDEN, HIDDEN),
            k: lin(&map, &format!("{ab}.k_proj"), N_KV * HEAD_DIM, HIDDEN),
            v: lin(&map, &format!("{ab}.v_proj"), N_KV * HEAD_DIM, HIDDEN),
            o: lin(&map, &format!("{ab}.o_proj"), HIDDEN, HIDDEN),
            mlp: DitMlp {
                gate: lin(&map, &format!("{mb}.gate_proj"), FFN, HIDDEN),
                up: lin(&map, &format!("{mb}.up_proj"), FFN, HIDDEN),
                down: lin(&map, &format!("{mb}.down_proj"), HIDDEN, FFN),
            },
            adaln: lin(&map, &format!("{lb}.speaker_adaln"), 6 * HIDDEN, SPK),
        });
    }
    DitW {
        in_proj: lin(&map, "in_proj", HIDDEN, LATENT),
        cond_proj: lin(&map, "cond_proj", HIDDEN, LATENT),
        out_proj: lin(&map, "out_proj", LATENT, HIDDEN),
        time_mlp1: lin(&map, "time_mlp.linear_1", HIDDEN, HIDDEN),
        time_mlp2: lin(&map, "time_mlp.linear_2", HIDDEN, HIDDEN),
        delta_mlp1: lin(&map, "delta_time_mlp.linear_1", HIDDEN, HIDDEN),
        delta_mlp2: lin(&map, "delta_time_mlp.linear_2", HIDDEN, HIDDEN),
        step_mlp1: lin(&map, "step_size_embedding.linear_1", HIDDEN, HIDDEN),
        step_mlp2: lin(&map, "step_size_embedding.linear_2", HIDDEN, HIDDEN),
        cfg_emb0: lin(&map, "cfg_strength_embedding.0", HIDDEN, 1),
        cfg_emb2: lin(&map, "cfg_strength_embedding.2", HIDDEN, HIDDEN),
        layers: [layers.remove(0), layers.remove(0), layers.remove(0), layers.remove(0)],
        final_norm: norm_w(&map, "decoder.norm.weight", HIDDEN),
    }
}
