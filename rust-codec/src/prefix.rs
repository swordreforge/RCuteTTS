//! Voice-clone prefix assembly (pure Rust reference path).
//!
//! Mirrors `processor._reference_prompt_segments` + `SegmentManager.fuse` +
//! `prepare_input_embeds` for the distill voice_clone branch (5 segments,
//! total 71 for the default reference):
//!   text(prompt1) + speaker_linear(1) + text("<|im_end|>\\n<|im_start|>")
//!   + speech(23 patches) + text(suffix)
//! Prompt strings are byte-exact copies of processor.py:88-113.
//! Reference chain: wav -> mono -> resample 24k (<=30s, repeat if <2s) ->
//! VAE-encode -> [T,64] -> patch [T/2,2,64] -> (x+bias)*scale ->
//! LocEnc+proj -> fill speech slots. Speaker slot: lm_speaker_linear
//! (bf16 [1024,256] -> fp32 at load) applied to the ECAPA embedding.
//! tts has no reference/speaker (see CLI); this module covers clone.

use crate::dit::DitLinear;
use crate::gemm::pack_a;
use crate::locenc::{locenc_embed, LocencW};
use crate::qwen::QwenW;
use crate::resample::resample;
use crate::simd::bf16_to_f32;
use crate::tok::PromptTokenizer;
use crate::vae_enc::{vae_encode, VaeEncW};
use safetensors::SafeTensors;

pub const P1: &str = "Transform the text into speech output, utilizing the distinct voice of the provided speech sample.\nvoice reference:\n<|im_start|>";
pub const P2: &str = "<|im_end|>\n<|im_start|>";
pub const SUFFIX_TAIL: &str = "<|endofprompt|>";

pub fn clone_suffix(target: &str) -> String {
    format!("<|im_end|>\ntext input:\n{target}\n{SUFFIX_TAIL}")
}

pub struct PrefixW {
    pub speaker_lin: DitLinear, // [1024, 256] bf16->fp32, no bias
}

pub fn load_prefix_weights(path: &std::path::Path) -> PrefixW {
    let bytes = std::fs::read(path).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let tv = st.tensor("lm_speaker_linear.weight").unwrap();
    assert_eq!(tv.shape(), &[1024, 256]);
    let raw = tv.data();
    let src: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let mut dst = vec![0.0f32; 1024 * 256];
    bf16_to_f32(&mut dst, &src);
    let g = pack_a(&dst, 1024, 256);
    let bp = vec![0.0f32; g.rows];
    PrefixW { speaker_lin: DitLinear { w: g, b: bp, out_dim: 1024 } }
}

/// Reference wave (mono f32 + source sr) -> 24k reference branch.
/// Mirrors prepare_reference_audio EXACTLY: resample the (30s-capped) wave,
/// crop 30s, then FULL repeats while shorter than 2s (no truncation).
pub fn reference_branch_24k(wave: &[f32], sr: usize) -> Vec<f32> {
    if wave.is_empty() {
        panic!("reference audio is empty");
    }
    let mut w = if sr == 24000 { wave.to_vec() } else { resample(wave, sr, 24000) };
    w.truncate(30 * 24000);
    while w.len() < 2 * 24000 {
        let cur = w.clone();
        w.extend_from_slice(&cur);
    }
    w
}

/// Speaker branch: FIRST 8s of source, resample to 16k, crop, full repeats.
pub fn speaker_branch_16k(wave: &[f32], sr: usize) -> Vec<f32> {
    if wave.is_empty() {
        panic!("reference audio is empty");
    }
    let src: Vec<f32> = wave.iter().copied().take(8 * sr).collect();
    let mut w = if sr == 16000 { src } else { resample(&src, sr, 16000) };
    w.truncate(8 * 16000);
    while w.len() < 2 * 16000 {
        let cur = w.clone();
        w.extend_from_slice(&cur);
    }
    w
}

/// Average channels to mono.
pub fn to_mono(pcm: &[f32], channels: usize) -> Vec<f32> {
    assert_eq!(pcm.len() % channels, 0);
    let n = pcm.len() / channels;
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let mut s = 0.0f32;
        for c in 0..channels {
            s += pcm[i * channels + c];
        }
        out[i] = s / channels as f32;
    }
    out
}

/// Full clone prefix: returns (embeds [T,1024] row-major, T).
/// speech_frames: VAE-encoded reference [F,64]; spk: ECAPA [256].
/// scale/bias: checkpoint speech scalars (apply (x+bias)*scale like
/// forward_speech_features).
pub fn clone_prefix(
    tok: &PromptTokenizer,
    qw: &QwenW,
    lw: &LocencW,
    pw: &PrefixW,
    target_text: &str,
    speech_frames: &[f32],
    nframes: usize,
    spk: &[f32],
    scale: f32,
    bias: f32,
    nth: usize,
) -> (Vec<f32>, usize) {
    // tokenize pieces
    let ids1 = tok.encode(P1);
    let ids2 = tok.encode(P2);
    let ids3 = tok.encode(&clone_suffix(target_text));
    // patch speech frames (audio_patch_size=2, zero-pad odd)
    let mut fr = speech_frames.to_vec();
    let nf = nframes;
    let rem = nf % 2;
    let np = if rem == 0 { nf } else { nf + 1 };
    fr.resize(np * 64, 0.0);
    // scale (x+bias)*scale
    for v in fr.iter_mut() {
        *v = (*v + bias) * scale;
    }
    // locenc over [1, np/2, 2, 64] -> [np/2, 1024]
    let speech_emb = locenc_embed(lw, &fr, 1, np / 2, nth);
    // speaker slot
    let mut spk_proj = vec![0.0f32; 1024];
    {
        let mut c = vec![0.0f32; pw.speaker_lin.w.rows];
        crate::gemm::sgemm_bias(&pw.speaker_lin.w, spk, 1, &pw.speaker_lin.b, &mut c, 1);
        spk_proj.copy_from_slice(&c[..1024]);
    }
    // assemble: ids1 + [0](speaker slot) + ids2 + [0]*np/2 (audio pads) + ids3
    let mut ids: Vec<i64> = Vec::new();
    ids.extend(ids1.iter());
    let spk_pos = ids.len();
    ids.push(0);
    ids.extend(ids2.iter());
    let aud_pos = ids.len();
    ids.extend(std::iter::repeat(0).take(np / 2));
    ids.extend(ids3.iter());
    let t = ids.len();
    // embed lookup
    let mut embeds = vec![0.0f32; t * 1024];
    for (s, &id) in ids.iter().enumerate() {
        embeds[s * 1024..(s + 1) * 1024]
            .copy_from_slice(&qw.embed[id as usize * 1024..(id as usize + 1) * 1024]);
    }
    // fill speech slots + speaker slot
    for p in 0..np / 2 {
        embeds[(aud_pos + p) * 1024..(aud_pos + p + 1) * 1024]
            .copy_from_slice(&speech_emb[p * 1024..(p + 1) * 1024]);
    }
    embeds[spk_pos * 1024..(spk_pos + 1) * 1024].copy_from_slice(&spk_proj);
    (embeds, t)
}

/// Reference features: resampled 24k wave -> VAE-encode -> ([F,64], F).
pub fn reference_features(ve: &VaeEncW, wave24k: &[f32], nth: usize) -> (Vec<f32>, usize) {
    let mu = vae_encode(ve, wave24k, nth);
    let f = mu.len() / 64;
    (mu, f)
}
