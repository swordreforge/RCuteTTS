//! Qwen3 forward differential test vs torch (`scripts/dump_qwen_vectors.py`).
//! Prefill (T=48) + 3 chained decodes. Tight gate vs fp32 (< 1e-2);
//! bf16 production gap (2.8e-1) is recorded, not gated.
use cutetts_codec::qwen::{decode_step, load_qwen_weights, prefill, QwenCache};
use ndarray_npy::read_npy;
use std::path::PathBuf;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn load_flat(name: &str) -> Vec<f32> {
    let p = manifest().join(format!("testdata/{name}.npy"));
    if let Ok(a) = read_npy::<_, ndarray::Array3<f32>>(p.clone()) {
        return a.into_raw_vec_and_offset().0;
    }
    read_npy::<_, ndarray::Array2<f32>>(p).unwrap().into_raw_vec_and_offset().0
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn qwen_prefill_decode_chain() {
    let wpath = manifest()
        .join("..")
        .join("model/CuteTTS-distill/weights/tts/model.safetensors");
    if !wpath.is_file() {
        eprintln!("skip: weights not found at {}", wpath.display());
        return;
    }
    let w = load_qwen_weights(&wpath);
    let mut cache = QwenCache::empty();
    // prefill
    let pre = load_flat("qwen_p_fp32");
    let ref_p = load_flat("qwen_p_out_fp32");
    assert_eq!(pre.len(), 48 * 1024);
    let t0 = std::time::Instant::now();
    let h = prefill(&w, &pre, 48, 0, &mut cache, 1);
    println!("prefill T=48: {:.2}s", t0.elapsed().as_secs_f32());
    let e = max_err(&h, &ref_p);
    println!("prefill err={e:.2e}");
    assert!(e < 1e-3, "prefill max_err={e}");
    // chained decodes
    for i in 0..3 {
        let x = load_flat(&format!("qwen_d{i}_fp32"));
        let refr = load_flat(&format!("qwen_d{i}_out_fp32"));
        let t1 = std::time::Instant::now();
        let out = decode_step(&w, &x, 48 + i, &mut cache);
        println!("decode[{i}]: {:.3}s", t1.elapsed().as_secs_f32());
        let e = max_err(&out, &refr);
        println!("decode[{i}] err={e:.2e}");
        assert!(e < 1e-3, "decode[{i}] max_err={e}");
    }
    println!("Qwen chain gate passed");
}
