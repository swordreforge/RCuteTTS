//! VAE-encoder differential test vs torch (`scripts/dump_encode_vectors.py`).
//! mu only (posterior mode); gate 1e-3 first, tighten after measuring.
use cutetts_codec::vae_enc::{load_vae_enc_weights, vae_encode};
use ndarray_npy::read_npy;
use std::path::PathBuf;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn load1(name: &str) -> Vec<f32> {
    let p = manifest().join(format!("testdata/{name}.npy"));
    if let Ok(a) = read_npy::<_, ndarray::Array2<f32>>(p.clone()) {
        return a.into_raw_vec_and_offset().0;
    }
    read_npy::<_, ndarray::Array1<f32>>(p).unwrap().to_vec()
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn vae_encode_vectors() {
    let wpath = manifest().join("../model/CuteTTS/weights/audio_vae/model.safetensors");
    if !wpath.is_file() {
        eprintln!("skip: weights missing");
        return;
    }
    let t0 = std::time::Instant::now();
    let w = load_vae_enc_weights(&wpath);
    println!("encoder weights loaded in {:.1}s", t0.elapsed().as_secs_f32());
    let nth = cutetts_codec::conv::default_threads();
    let mut worst = 0.0f32;
    for tag in ["enc_00", "enc_01", "enc_02", "enc_ref"] {
        let wave = load1(&format!("{tag}_wav"));
        let refr = load1(&format!("{tag}_mu"));
        let t1 = std::time::Instant::now();
        let mu = vae_encode(&w, &wave, nth);
        let dt = t1.elapsed().as_secs_f32();
        assert_eq!(mu.len(), refr.len(), "{tag} len");
        let e = max_err(&mu, &refr);
        worst = worst.max(e);
        println!("{tag}: frames={} err={e:.2e} time={dt:.2}s", refr.len() / 64);
    }
    println!("VAE-ENC gate passed, worst={worst:.2e}");
    assert!(worst < 1e-3, "worst {worst:.2e}");
}
