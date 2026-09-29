//! End-to-end offline tts ring (distill): prefill -> AR loop -> whole VAE decode.
//!
//! Wiring mirrors `_naive_ar_infer_impl` (generation.py) for the distill
//! single-branch offline path:
//!   prefill(prefix bf16->fp32) -> loop { stop? -> DiT(plain) -> scale ->
//!   LocEnc feedback -> LM decode } -> concat scaled latents -> VAE decode.
//! Production runs LM/LocEnc in bf16; Rust runs everything fp32, so stage
//! gates below allow the measured dtype drift (see dump output).

use crate::dit::{euler_sample, load_dit_weights, DitW};
use crate::locenc::{load_locenc_weights, locenc_embed, LocencW};
use crate::qwen::{decode_step, load_qwen_weights, prefill, QwenCache, QwenW};
use crate::weights::load_decoder_weights;
use crate::weights::DecoderW;
use safetensors::SafeTensors;
use std::collections::HashMap;

pub struct E2EW {
    pub stop_w: Vec<f32>, // [2, 1024] row-major
    pub stop_b: Vec<f32>, // [2]
    pub scale: f32,
    pub bias: f32,
}

pub struct AllW {
    pub qwen: QwenW,
    pub dit: DitW,
    pub locenc: LocencW,
    pub vae: DecoderW,
    pub e2e: E2EW,
}

fn f32_of(tv: &safetensors::tensor::TensorView<'_>) -> Vec<f32> {
    match tv.dtype() {
        safetensors::Dtype::F32 => tv
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        safetensors::Dtype::BF16 => {
            let mut dst = vec![0.0f32; tv.data().len() / 2];
            let src: Vec<u16> = tv
                .data()
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            crate::simd::bf16_to_f32(&mut dst, &src);
            dst
        }
        d => panic!("unexpected dtype {d:?}"),
    }
}

pub fn load_e2e_weights(path: &std::path::Path) -> E2EW {
    let bytes = std::fs::read(path).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let mut map: HashMap<String, Vec<f32>> = HashMap::new();
    for name in st.names() {
        if name.starts_with("stop_predictor.") || name.starts_with("speech_") {
            let tv = st.tensor(name).unwrap();
            map.insert(name.to_string(), f32_of(&tv));
        }
    }
    E2EW {
        stop_w: map["stop_predictor.weight"].clone(),
        stop_b: map["stop_predictor.bias"].clone(),
        scale: map["speech_scaling_factor"][0],
        bias: map["speech_bias_factor"][0],
    }
}

pub fn load_all(tts_path: &std::path::Path, vae_path: &std::path::Path) -> AllW {
    AllW {
        qwen: load_qwen_weights(tts_path),
        dit: load_dit_weights(tts_path),
        locenc: load_locenc_weights(tts_path),
        vae: load_decoder_weights(vae_path),
        e2e: load_e2e_weights(tts_path),
    }
}

/// stop logits [2] for a hidden state (argmax==1 means stop).
pub fn stop_logits(e: &E2EW, h: &[f32]) -> [f32; 2] {
    assert_eq!(h.len(), 1024);
    let mut out = [e.stop_b[0], e.stop_b[1]];
    for r in 0..2 {
        for (i, &hv) in h.iter().enumerate() {
            out[r] += e.stop_w[r * 1024 + i] * hv;
        }
    }
    out
}

/// One AR step's DiT + scale, returning (pred_raw[128], pred_scaled[128]).
pub fn dit_step(
    w: &AllW,
    x0: &[f32],
    z: &[f32],
    cond: &[f32],
    nth: usize,
) -> (Vec<f32>, Vec<f32>) {
    let pred = euler_sample(&w.dit, x0, z, cond, None, 4, 2.0, nth);
    let scaled: Vec<f32> = pred.iter().map(|&v| v / w.e2e.scale - w.e2e.bias).collect();
    (pred, scaled)
}
