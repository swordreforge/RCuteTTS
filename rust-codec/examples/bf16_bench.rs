//! bf16->f32 throughput at LM scale (127M bf16 = Qwen3-7L backbone).
//! Run: cargo run --release --example bf16_bench
use cutetts_codec::simd::bf16_to_f32;
use std::time::Instant;

fn main() {
    let n = 127_000_000usize; // ~Qwen3 backbone bf16 count
    let mut src = vec![0u16; n];
    for (i, v) in src.iter_mut().enumerate() {
        // representative weight bits (finite normals around 0)
        *v = (0x3C00u16.wrapping_add((i & 0xFF) as u16)) ^ ((i >> 8) as u16 & 0x7F);
    }
    let mut dst = vec![0.0f32; n];
    // warmup
    bf16_to_f32(&mut dst, &src);
    let reps = 5;
    let t0 = Instant::now();
    for _ in 0..reps {
        bf16_to_f32(&mut dst, &src);
    }
    let s = t0.elapsed().as_secs_f32() / reps as f32;
    let gb = (n * 2 + n * 4) as f64 / 1e9;
    println!("bf16->f32 x{n}: {s:.3}s = {gb_per_s:.1} GB/s (memcpy-bound)", gb_per_s = gb / s as f64);
    // spot check vs shift identity
    assert_eq!(dst[0].to_bits(), (src[0] as u32) << 16);
    assert_eq!(dst[n - 1].to_bits(), (src[n - 1] as u32) << 16);
    println!("spot check ok");
}
