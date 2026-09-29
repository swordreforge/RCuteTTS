//! DiT _predict differential test vs torch (`scripts/dump_dit_vectors.py`).
//! Gate: max abs err < 1e-3.
use cutetts_codec::dit::{euler_sample, load_dit_weights, predict};
use ndarray_npy::read_npy;
use std::path::PathBuf;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn load_vec(name: &str) -> Vec<f32> {
    let p = manifest().join(format!("testdata/{name}.npy"));
    // npy may be 1D/2D/3D; read raw via 1D flatten view
    if let Ok(a) = read_npy::<_, ndarray::Array1<f32>>(p.clone()) {
        return a.to_vec();
    }
    if let Ok(a) = read_npy::<_, ndarray::Array2<f32>>(p.clone()) {
        return a.into_raw_vec_and_offset().0;
    }
    let a = read_npy::<_, ndarray::Array3<f32>>(p).unwrap();
    a.into_raw_vec_and_offset().0
}

#[test]
fn dit_predict_3_vectors() {
    let wpath = manifest()
        .join("..")
        .join("model/CuteTTS-distill/weights/tts/model.safetensors");
    if !wpath.is_file() {
        eprintln!("skip: weights not found at {}", wpath.display());
        return;
    }
    let w = load_dit_weights(&wpath);
    let mut worst = 0.0f32;
    for i in 0..3 {
        let x = load_vec(&format!("dit_p{i:02}_x"));
        let t = load_vec(&format!("dit_p{i:02}_t"));
        let z = load_vec(&format!("dit_p{i:02}_z"));
        let cond = load_vec(&format!("dit_p{i:02}_cond"));
        let spk = load_vec(&format!("dit_p{i:02}_spk"));
        let refr = load_vec(&format!("dit_p{i:02}_out"));
        assert_eq!((x.len(), z.len(), cond.len(), spk.len(), refr.len()), (128, 1024, 128, 256, 128));
        let t0 = std::time::Instant::now();
        let v = predict(&w, &x, t[0], &z, &cond, 0.25, Some(&spk), 2.0, 1);
        let dt = t0.elapsed().as_secs_f32();
        assert_eq!(v.len(), refr.len());
        let e: f32 = v.iter().zip(refr.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        worst = worst.max(e);
        println!("dit_p{i:02}: err={e:.2e} time={dt:.2}s");
        assert!(e < 1e-3, "dit_p{i:02} max_err={e}");
    }
    println!("DiT predict gate passed, worst err={worst:.2e}");
}

#[test]
fn dit_predict_plain_no_speaker() {
    // tts path: speaker_embedding=None -> plain residuals (no modulate/gate).
    let wpath = manifest()
        .join("..")
        .join("model/CuteTTS-distill/weights/tts/model.safetensors");
    if !wpath.is_file() {
        eprintln!("skip: weights not found at {}", wpath.display());
        return;
    }
    let w = load_dit_weights(&wpath);
    let x = load_vec("dit_nospk_x");
    let t = vec![0.25f32];
    let z = load_vec("dit_nospk_z");
    let cond = load_vec("dit_nospk_cond");
    let refr = load_vec("dit_nospk_out");
    let v = predict(&w, &x, t[0], &z, &cond, 0.25, None, 2.0, 1);
    let e: f32 = v.iter().zip(refr.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    println!("dit_nospk: err={e:.2e}");
    assert!(e < 1e-3, "dit_nospk max_err={e}");
}

#[test]
fn dit_euler_sample_2_vectors() {
    let wpath = manifest()
        .join("..")
        .join("model/CuteTTS-distill/weights/tts/model.safetensors");
    if !wpath.is_file() {
        eprintln!("skip: weights not found at {}", wpath.display());
        return;
    }
    let w = load_dit_weights(&wpath);
    let mut worst = 0.0f32;
    for i in 0..2 {
        let z = load_vec(&format!("dit_s{i:02}_z"));
        let cond = load_vec(&format!("dit_s{i:02}_cond"));
        let spk = load_vec(&format!("dit_s{i:02}_spk"));
        let x0 = load_vec(&format!("dit_s{i:02}_x0"));
        let refr = load_vec(&format!("dit_s{i:02}_out"));
        let t0 = std::time::Instant::now();
        let out = euler_sample(&w, &x0, &z, &cond, Some(&spk), 4, 2.0, 1);
        let dt = t0.elapsed().as_secs_f32();
        assert_eq!(out.len(), refr.len());
        let e: f32 = out.iter().zip(refr.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        worst = worst.max(e);
        println!("dit_s{i:02}: err={e:.2e} time={dt:.2}s");
        assert!(e < 1e-3, "dit_s{i:02} max_err={e}");
    }
    println!("DiT euler gate passed, worst err={worst:.2e}");
}
