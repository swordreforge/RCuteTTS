//! Qwen prefill/decode bench vs torch.
//! Run: cargo run --release --example qwen_bench (from rust-codec/)
use cutetts_codec::conv::default_threads;
use cutetts_codec::qwen::{decode_step, load_qwen_weights, prefill, QwenCache};
use std::time::Instant;

fn load_flat(name: &str) -> Vec<f32> {
    let p = format!("testdata/{name}.npy");
    if let Ok(a) = ndarray_npy::read_npy::<_, ndarray::Array3<f32>>(p.clone()) {
        return a.into_raw_vec_and_offset().0;
    }
    ndarray_npy::read_npy::<_, ndarray::Array2<f32>>(p).unwrap().into_raw_vec_and_offset().0
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn main() {
    let w = load_qwen_weights(std::path::Path::new(
        "../model/CuteTTS-distill/weights/tts/model.safetensors",
    ));
    let nth = default_threads();
    let mut cache = QwenCache::empty();
    let pre = load_flat("qwen_p_fp32");
    let ref_p = load_flat("qwen_p_out_fp32");
    prefill(&w, &pre, 48, 0, &mut cache, nth); // warmup
    // cold-vs-warm check + T=71 scaling probe (CLI prefill anomaly)
    let mut c71 = QwenCache::empty();
    let mut pre71 = vec![0.0f32; 71 * 1024];
    pre71[..48 * 1024].copy_from_slice(&pre);
    let t71 = std::time::Instant::now();
    prefill(&w, &pre71, 71, 0, &mut c71, nth);
    println!("prefill T=71: {:.3}s", t71.elapsed().as_secs_f32());
    let reps = 5;
    let t0 = Instant::now();
    for _ in 0..reps {
        let mut c = QwenCache::empty();
        prefill(&w, &pre, 48, 0, &mut c, nth);
    }
    println!("prefill T=48: {:.3}s (threads={nth})", t0.elapsed().as_secs_f32() / reps as f32);
    // chained decodes on the warmed cache, then fresh caches for timing
    let mut ts = vec![];
    for r in 0..reps {
        let mut c = QwenCache::empty();
        prefill(&w, &pre, 48, 0, &mut c, nth);
        let t1 = Instant::now();
        for i in 0..3 {
            let x = load_flat(&format!("qwen_d{i}_fp32"));
            let out = decode_step(&w, &x, 48 + i, &mut c);
            if r == 0 && i == 2 {
                let refr = load_flat("qwen_d2_out_fp32");
                println!("decode err={:.2e}", max_err(&out, &refr));
            }
        }
        ts.push(t1.elapsed().as_secs_f32() / 3.0);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("decode median: {:.1}ms", ts[reps / 2] * 1000.0);
    let _ = ref_p;
}
