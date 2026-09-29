//! LocEnc differential test vs torch (`scripts/dump_locenc_vectors.py`).
//! Tight gate vs fp32 (< 1e-3); bf16 production gap is recorded, not gated
//! (measured 3~8e-2 abs — see dump output).
use cutetts_codec::locenc::{load_locenc_weights, locenc_embed};
use ndarray_npy::read_npy;
use std::path::PathBuf;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn load_flat(name: &str) -> (Vec<f32>, Vec<usize>) {
    let p = manifest().join(format!("testdata/{name}.npy"));
    if let Ok(a) = read_npy::<_, ndarray::Array4<f32>>(p.clone()) {
        let s = a.shape().to_vec();
        return (a.into_raw_vec_and_offset().0, s);
    }
    if let Ok(a) = read_npy::<_, ndarray::Array3<f32>>(p.clone()) {
        let s = a.shape().to_vec();
        return (a.into_raw_vec_and_offset().0, s);
    }
    let a = read_npy::<_, ndarray::Array2<f32>>(p).unwrap();
    let s = a.shape().to_vec();
    (a.into_raw_vec_and_offset().0, s)
}

#[test]
fn locenc_3_vectors() {
    let wpath = manifest()
        .join("..")
        .join("model/CuteTTS-distill/weights/tts/model.safetensors");
    if !wpath.is_file() {
        eprintln!("skip: weights not found at {}", wpath.display());
        return;
    }
    let w = load_locenc_weights(&wpath);
    let mut worst = 0.0f32;
    for tag in ["b", "c", "d"] {
        let (x, xs) = load_flat(&format!("loc_{tag}_in"));
        let (ref32, rs) = load_flat(&format!("loc_{tag}_fp32"));
        let (b, t) = (xs[0], xs[1]);
        assert_eq!(xs[2..], vec![2, 64]);
        assert_eq!(rs, vec![b, t, 1024]);
        let t0 = std::time::Instant::now();
        let out = locenc_embed(&w, &x, b, t, 1);
        let dt = t0.elapsed().as_secs_f32();
        assert_eq!(out.len(), ref32.len());
        let e: f32 = out.iter().zip(ref32.iter()).map(|(a, c)| (a - c).abs()).fold(0.0, f32::max);
        worst = worst.max(e);
        println!("loc_{tag}: fp32 err={e:.2e} time={dt:.2}s");
        assert!(e < 1e-3, "loc_{tag} max_err={e}");
    }
    println!("LocEnc gate passed, worst err={worst:.2e}");
}
