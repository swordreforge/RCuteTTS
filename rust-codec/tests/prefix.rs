//! Clone prefix parity vs torch (`scripts/dump_clone_prefix.py`).
//! Per-region gates: text rows bitwise (pure lookup), speech rows loose
//! (torch runs locenc bf16: floor ~5e-2), speaker slot small (linear bf16).
use cutetts_codec::locenc::load_locenc_weights;
use cutetts_codec::prefix::{clone_prefix, clone_suffix, load_prefix_weights, reference_features, P1, P2};
use cutetts_codec::qwen::load_qwen_weights;
use cutetts_codec::tok::PromptTokenizer;
use cutetts_codec::vae_enc::load_vae_enc_weights;
use ndarray_npy::read_npy;
use std::path::PathBuf;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn load_ids(name: &str) -> Vec<i64> {
    let p = manifest().join(format!("testdata/{name}.npy"));
    read_npy::<_, ndarray::Array1<i64>>(p).unwrap().to_vec()
}

fn load_flat(name: &str) -> Vec<f32> {
    let p = manifest().join(format!("testdata/{name}.npy"));
    if let Ok(a) = read_npy::<_, ndarray::Array3<f32>>(p.clone()) {
        return a.into_raw_vec_and_offset().0;
    }
    if let Ok(a) = read_npy::<_, ndarray::Array2<f32>>(p.clone()) {
        return a.into_raw_vec_and_offset().0;
    }
    read_npy::<_, ndarray::Array1<f32>>(p).unwrap().to_vec()
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn clone_prefix_matches() {
    let tts = manifest().join("../model/CuteTTS-distill/weights/tts/model.safetensors");
    let vae = manifest().join("../model/CuteTTS/weights/audio_vae/model.safetensors");
    if !tts.is_file() || !vae.is_file() {
        eprintln!("skip: weights missing");
        return;
    }
    let nth = cutetts_codec::conv::default_threads();
    let tok = PromptTokenizer::load(&manifest().join("../model/CuteTTS-distill/tokenizer/tokenizer.model"));
    // piece ids vs fixtures
    for (tag, text) in [("cp1", P1.to_string()), ("cp2", P2.to_string()), ("cp3", clone_suffix("Hello world, this is a test."))] {
        let refr = load_ids(&format!("clone_{tag}_ids"));
        let got = tok.encode(&text);
        assert_eq!(got, refr, "{tag} ids");
    }
    println!("clone pieces exact match");
    let qw = load_qwen_weights(&tts);
    let lw = load_locenc_weights(&tts);
    let ve = load_vae_enc_weights(&vae);
    let pw = load_prefix_weights(&tts);
    let e2e = cutetts_codec::e2e::load_e2e_weights(&tts);
    // reference chain from the 24k wave fixture (skip file reading here)
    let wave = load_flat("enc_ref_wav");
    let (feats, nf) = reference_features(&ve, &wave, nth);
    let reffeats = load_flat("clone_ref_feats");
    assert_eq!(feats.len(), reffeats.len());
    let ef = max_err(&feats, &reffeats);
    println!("reference features err={ef:.2e} (frames={nf})");
    assert!(ef < 1e-3, "ref feats {ef:.2e}");
    // full prefix with fixture speaker embedding
    let spk = load_flat("spk_emb");
    let (embeds, t) = clone_prefix(&tok, &qw, &lw, &pw, "Hello world, this is a test.", &feats, nf, &spk, e2e.scale, e2e.bias, nth);
    assert_eq!(t, 71);
    let refr = load_flat("clone_prefix");
    assert_eq!(embeds.len(), refr.len());
    // per-region: text rows bitwise; speech/speaker rows bf16-tolerant
    let mut e_text = 0.0f32;
    let mut e_other = 0.0f32;
    // layout: [ids1(31), spk(1), ids2(2), speech(23), ids3(14)] = 71
    for s in 0..t {
        let e = max_err(&embeds[s * 1024..(s + 1) * 1024], &refr[s * 1024..(s + 1) * 1024]);
        if s < 31 || (32..34).contains(&s) || s >= 57 {
            e_text = e_text.max(e);
        } else {
            e_other = e_other.max(e);
        }
    }
    println!("prefix text rows err={e_text:.2e} speech/speaker rows err={e_other:.2e}");
    assert!(e_text == 0.0, "text rows must be bitwise");
    assert!(e_other < 0.3, "speech/speaker rows {e_other:.2e}");
    println!("CLONE PREFIX gate passed");
}
