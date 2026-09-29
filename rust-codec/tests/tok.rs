//! Tokenizer parity vs HF (`scripts/dump_prefix_vectors.py`).
//! All four texts must match id-for-id (no tolerance: tokenizers are exact).
use cutetts_codec::tok::PromptTokenizer;
use ndarray_npy::read_npy;
use std::path::PathBuf;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn load_ids(name: &str) -> Vec<i64> {
    let p = manifest().join(format!("testdata/{name}.npy"));
    read_npy::<_, ndarray::Array1<i64>>(p).unwrap().to_vec()
}

#[test]
fn tokenizer_matches_hf() {
    let tok = PromptTokenizer::load(
        &manifest().join("../model/CuteTTS-distill/tokenizer/tokenizer.model"),
    );
    assert_eq!(tok.piece_size(), 16384);
    for tag in ["e2e", "zh", "tricky", "short"] {
        let prompt =
            std::fs::read_to_string(manifest().join(format!("testdata/tok_{tag}_prompt.txt"))).unwrap();
        let refr = load_ids(&format!("tok_{tag}_ids"));
        let got = tok.encode(&prompt);
        assert_eq!(got.len(), refr.len(), "{tag} len");
        assert_eq!(got, refr, "{tag} ids differ");
        println!("tok_{tag}: {len} ids exact match", len = got.len());
    }
    println!("TOKENIZER gate passed");
}

#[test]
fn prefix_embeds_match() {
    // ids -> embed lookup == torch prefix embeds (bit-exact gather;
    // torch ran bf16, table values convert exactly).
    let wpath = manifest().join("../model/CuteTTS-distill/weights/tts/model.safetensors");
    if !wpath.is_file() {
        eprintln!("skip: weights missing");
        return;
    }
    let w = cutetts_codec::qwen::load_qwen_weights(&wpath);
    let tok = PromptTokenizer::load(
        &manifest().join("../model/CuteTTS-distill/tokenizer/tokenizer.model"),
    );
    let prompt =
        std::fs::read_to_string(manifest().join("testdata/tok_e2e_prompt.txt")).unwrap();
    let ids = tok.encode(&prompt);
    let got = cutetts_codec::qwen::embed_lookup(&w, &ids);
    let p = manifest().join("testdata/tok_e2e_prefix.npy");
    let refr: Vec<f32> = read_npy::<_, ndarray::Array3<f32>>(p).unwrap().into_raw_vec_and_offset().0;
    assert_eq!(got.len(), refr.len());
    let e: f32 = got.iter().zip(refr.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    println!("prefix embeds err={e:.2e}");
    assert!(e == 0.0, "prefix embeds must be bitwise, got {e:.2e}");
}
