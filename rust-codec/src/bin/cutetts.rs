//! Pure-Rust tts CLI (distill, offline): text -> wav, no Python.
//!
//! Usage (from rust-codec/):
//!   cutetts --text "Hello world." --output out.wav [--seed 42] [--max-steps 750]
//!   cutetts --model-dir ../model/CuteTTS --text "..." --output out.wav
//!
//! Pipeline: tokenize -> embed lookup -> Qwen prefill -> AR loop
//! (LM decode -> stop? -> DiT 4-step (own xorshift RNG) -> scale ->
//! LocEnc feedback) -> concat latents -> VAE whole decode -> 24k i16 wav.
//! x0 RNG is our own (torch Philox not replicated — chaos makes x0-exactness
//! pointless; trajectory validated bit-exact given same x0 in tests).

use cutetts_codec::conv::default_threads;
use cutetts_codec::decode::decode_nth;
use cutetts_codec::dit::euler_sample;
use cutetts_codec::e2e::{load_all, stop_logits};
use cutetts_codec::locenc::locenc_embed;
use cutetts_codec::qwen::{decode_step, embed_lookup, prefill, QwenCache};
use cutetts_codec::tok::PromptTokenizer;
use std::path::PathBuf;
use std::time::Instant;

/// xorshift64* + Box-Muller normals (deterministic, seeded).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn uniform(&mut self) -> f64 {
        // (0, 1]: never 0 (log-safe). Top 53 bits of xorshift output.
        const DIV: f64 = (1u64 << 53) as f64;
        1.0 - ((self.next_u64() >> 11) as f64) / DIV
    }

    fn normal(&mut self) -> f32 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() as f32 * (2.0 * std::f64::consts::PI * u2).cos() as f32
    }

    fn normals(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.normal()).collect()
    }
}

fn usage() -> ! {
    eprintln!("usage: cutetts --text TEXT --output OUT.wav [--model-dir DIR] [--seed N] [--max-steps N]");
    std::process::exit(2);
}

fn arg_val(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}

fn write_wav_16(path: &std::path::Path, samples: &[f32], sample_rate: u32) {    let mut data = Vec::with_capacity(44 + samples.len() * 2);
    let n = samples.len() as u32;
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&(36 + n * 2).to_le_bytes());
    data.extend_from_slice(b"WAVEfmt ");
    data.extend_from_slice(&16u32.to_le_bytes());
    data.extend_from_slice(&1u16.to_le_bytes());
    data.extend_from_slice(&1u16.to_le_bytes());
    data.extend_from_slice(&sample_rate.to_le_bytes());
    data.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    data.extend_from_slice(&2u16.to_le_bytes());
    data.extend_from_slice(&16u16.to_le_bytes());
    data.extend_from_slice(b"data");
    data.extend_from_slice(&(n * 2).to_le_bytes());
    for &v in samples {
        let s = (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        data.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, data).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = arg_val(&args, "--text").unwrap_or_else(|| usage());
    let out = arg_val(&args, "--output").unwrap_or_else(|| usage());
    let model_dir = arg_val(&args, "--model-dir").unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../model/CuteTTS-distill")
            .to_string_lossy()
            .into_owned()
    });
    let seed: u64 = arg_val(&args, "--seed").map(|v| v.parse().unwrap_or_else(|_| usage())).unwrap_or(42);
    let max_steps: usize = arg_val(&args, "--max-steps").map(|v| v.parse().unwrap_or_else(|_| usage())).unwrap_or(750);
    let nth = default_threads();
    let root = PathBuf::from(&model_dir);

    let t0 = Instant::now();
    let tok = PromptTokenizer::load(&root.join("tokenizer/tokenizer.model"));
    let w = load_all(
        &root.join("weights/tts/model.safetensors"),
        &root.join("weights/audio_vae/model.safetensors"),
    );
    println!("weights loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let t0 = Instant::now();
    let ids = tok.encode_tts(&text);
    println!("tokenized: {} ids", ids.len());
    let prefix = embed_lookup(&w.qwen, &ids);
    let tpre = ids.len();
    let mut cache = QwenCache::empty();
    let h = prefill(&w.qwen, &prefix, tpre, 0, &mut cache, nth);
    let mut last = h[(tpre - 1) * 1024..tpre * 1024].to_vec();
    println!("prefill T={tpre}: {:.2}s", t0.elapsed().as_secs_f32());

    let mut rng = Rng(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed });
    let mut cond = vec![0.0f32; 128]; // initial previous cond = zeros [1,2,64]
    let mut latents: Vec<f32> = Vec::new();
    let mut steps = 0;
    let t0 = Instant::now();
    loop {
        if steps >= max_steps {
            eprintln!("hit max steps {max_steps}");
            break;
        }
        let x0 = rng.normals(128);
        debug_assert!(x0.iter().all(|v| v.is_finite()), "x0 must be finite");
        let sl = stop_logits(&w.e2e, &last);
        if sl[1] > sl[0] {
            break;
        }
        let (pred, scaled) = {
            let p = euler_sample(&w.dit, &x0, &last, &cond, None, 4, 2.0, nth);
            let s: Vec<f32> = p.iter().map(|&v| v / w.e2e.scale - w.e2e.bias).collect();
            (p, s)
        };
        // next cond = UNSCALED pred ([2,64] flat, as returned)
        cond = pred;
        latents.extend_from_slice(&scaled);
        let fb = locenc_embed(&w.locenc, &scaled, 1, 1, nth);
        last = decode_step(&w.qwen, &fb, tpre + steps, &mut cache);
        steps += 1;
    }
    let dt_ar = t0.elapsed().as_secs_f32();
    println!("AR loop: {steps} steps in {dt_ar:.1}s ({:.1}ms/step)", dt_ar * 1000.0 / steps.max(1) as f32);

    let nframes = steps * 2;
    let mut frames = vec![0.0f32; 64 * nframes];
    for t in 0..nframes {
        for c in 0..64 {
            frames[c * nframes + t] = latents[t * 64 + c];
        }
    }
    let t0 = Instant::now();
    let wav = decode_nth(&w.vae, &frames, nframes, nth);
    println!("vae decode: {:.2}s", t0.elapsed().as_secs_f32());
    write_wav_16(PathBuf::from(&out).as_path(), &wav, 24000);
    let dur = wav.len() as f32 / 24000.0;
    println!("wrote {} ({dur:.2}s audio)", out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_standard_normal() {
        // Guards the Box-Muller domain bug (u1 in (0,1], never [1,2)):
        // all finite, mean ~0, var ~1, no NaN escape.
        let mut rng = Rng(42);
        let xs = rng.normals(100_000);
        assert!(xs.iter().all(|v| v.is_finite()));
        let mean = xs.iter().sum::<f32>() / xs.len() as f32;
        let var = xs.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / xs.len() as f32;
        println!("rng mean={mean:.4} var={var:.4}");
        assert!(mean.abs() < 0.02, "{mean}");
        assert!((var - 1.0).abs() < 0.03, "{var}");
        // determinism
        let mut a = Rng(7);
        let mut b = Rng(7);
        assert_eq!(a.normals(257), b.normals(257));
    }
}
