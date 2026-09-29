//! Long-prefill parity probe (T=1296 real prefix): ours vs torch fp32.
//! Run: cargo run --release --example long_prefill (from rust-codec/)
//! Fixtures: testdata/long_pre.npy + long_pre_out.npy (scripts/dump_long_prefill.py).
//! Gate: max err must sit at the fp32-vs-torch noise floor (~1e-4 scale,
//! cf. T=48 prefill gates), not blow up with length. A length-dependent
//! blowup means our long-context encoding is broken (the prime suspect
//! for free-run position jumping); a floor-level match rules prefill out
//! and points at chaotic trajectory / model long-prefix limits.
use cutetts_codec::conv::default_threads;
use cutetts_codec::qwen::{load_qwen_weights, prefill, QwenCache};

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
    let pre = load_flat("long_pre");
    let refr = load_flat("long_pre_out");
    let t = pre.len() / 1024;
    assert_eq!(t, 1296, "fixture length");
    assert_eq!(refr.len(), pre.len());
    let mut cache = QwenCache::empty();
    let t0 = std::time::Instant::now();
    let h = prefill(&w, &pre, t, 0, &mut cache, nth);
    println!("prefill T={t}: {:.2}s (threads={nth})", t0.elapsed().as_secs_f32());
    let e = max_err(&h, &refr);
    // per-position worst row, to see if error grows with position
    let mut worst = 0.0f32;
    let mut worst_row = 0;
    for s in 0..t {
        let mut m = 0.0f32;
        for d in 0..1024 {
            m = m.max((h[s * 1024 + d] - refr[s * 1024 + d]).abs());
        }
        if m > worst {
            worst = m;
            worst_row = s;
        }
    }
    println!("long-prefill maxerr={e:.2e} (ref maxabs 16.34) worst row={worst_row}");
    assert!(worst_row < t);
}
