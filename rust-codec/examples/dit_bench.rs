//! DiT euler_sample bench vs torch (`scripts/dump_dit_vectors.py` fixture).
//! Run: cargo run --release --example dit_bench
use cutetts_codec::conv::default_threads;
use cutetts_codec::dit::{euler_sample, load_dit_weights};
use std::time::Instant;

fn load_vec(name: &str) -> Vec<f32> {
    let p = format!("testdata/{name}.npy");
    if let Ok(a) = ndarray_npy::read_npy::<_, ndarray::Array3<f32>>(p.clone()) {
        return a.into_raw_vec_and_offset().0;
    }
    ndarray_npy::read_npy::<_, ndarray::Array2<f32>>(p).unwrap().into_raw_vec_and_offset().0
}

fn main() {
    let w = load_dit_weights(std::path::Path::new(
        "../model/CuteTTS-distill/weights/tts/model.safetensors",
    ));
    let nth = default_threads();
    let (z, cond, spk, x0) = (
        load_vec("dit_s00_z"),
        load_vec("dit_s00_cond"),
        load_vec("dit_s00_spk"),
        load_vec("dit_s00_x0"),
    );
    let refr = load_vec("dit_s00_out");
    euler_sample(&w, &x0, &z, &cond, &spk, 4, 2.0, nth); // warmup
    let reps = 5;
    let t0 = Instant::now();
    let mut out = vec![];
    for _ in 0..reps {
        out = euler_sample(&w, &x0, &z, &cond, &spk, 4, 2.0, nth);
    }
    let ms = t0.elapsed().as_secs_f32() * 1000.0 / reps as f32;
    let e: f32 = out.iter().zip(refr.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    println!("euler-4step: {ms:.1}ms (threads={nth}) max_err={e:.2e}");
}
