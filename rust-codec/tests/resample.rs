//! Resample parity vs torch (`scripts/dump_resample_vectors.py`).
//! Gate 1e-5 first (kernel f64-identical, accumulation same order).
use cutetts_codec::resample::resample;
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
fn resample_pairs() {
    let mut worst = 0.0f32;
    for (tag, orig, new) in [
        ("44100_24000", 44100, 24000),
        ("48000_24000", 48000, 24000),
        ("48000_16000", 48000, 16000),
        ("44100_16000", 44100, 16000),
        ("24000_16000", 24000, 16000),
        ("16000_24000", 16000, 24000),
        ("22050_24000", 22050, 24000),
        ("24000_24000", 24000, 24000),
    ] {
        let x = load1(&format!("rs_{tag}_in"));
        let refr = load1(&format!("rs_{tag}_out"));
        let t0 = std::time::Instant::now();
        let got = resample(&x, orig, new);
        let dt = t0.elapsed().as_secs_f32();
        assert_eq!(got.len(), refr.len(), "{tag} length");
        let e = max_err(&got, &refr);
        worst = worst.max(e);
        println!("rs_{tag}: len={} err={e:.2e} time={dt:.3}s", got.len());
        // Gate 5e-5 (~12x measured worst 4.2e-6; deterministic same-machine).
        assert!(e < 5e-5, "{tag} {e:.2e}");
    }
    println!("RESAMPLE gate passed, worst={worst:.2e}");
}
