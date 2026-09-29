//! M2: full-sentence decode vs torch (`scripts/dump_m2_vectors.py`).
//! Gate: max abs err < 1e-3 (accumulated f32 order differs over ~4 GFLOP).
//! Needs local weights: ../../model/CuteTTS/weights/audio_vae/model.safetensors
//! Run: cargo test --release --test m2_decode

use cutetts_codec::{decode::decode, weights::load_decoder_weights};
use ndarray_npy::read_npy;
use std::path::PathBuf;
use std::time::Instant;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn full_decode_10_vectors() {
    let wpath = manifest()
        .join("..")
        .join("model/CuteTTS/weights/audio_vae/model.safetensors");
    if !wpath.is_file() {
        eprintln!("skip: weights not found at {}", wpath.display());
        return;
    }
    let t0 = Instant::now();
    let w = load_decoder_weights(&wpath);
    println!("weights loaded+fused in {:.1}s", t0.elapsed().as_secs_f32());
    let mut worst = 0.0f32;
    for i in 0..10 {
        let lat: ndarray::Array2<f32> =
            read_npy(manifest().join(format!("testdata/m2_lat_{i:02}.npy"))).unwrap();
        let refr: ndarray::Array2<f32> =
            read_npy(manifest().join(format!("testdata/m2_ref_{i:02}.npy"))).unwrap();
        let refr = refr.as_slice().unwrap().to_vec();
        let frames = lat.shape()[1];
        let t1 = Instant::now();
        let wav = decode(&w, lat.as_slice().unwrap(), frames);
        let dt = t1.elapsed().as_secs_f32();
        assert_eq!(wav.len(), refr.len());
        let e: f32 = wav
            .iter()
            .zip(refr.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        worst = worst.max(e);
        println!("m2_{i:02}: frames={frames} samples={} err={e:.2e} time={dt:.1}s", wav.len());
        assert!(e < 1e-3, "m2_{i:02} max_err={e}");
    }
    println!("M2 gate passed, worst err={worst:.2e}");
}
