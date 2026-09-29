//! Pure-Rust tts CLI (distill, offline): text -> wav, no Python.
//!
//!   cutetts --text "Hello world." --output out.wav
//!   cutetts --mode voice_clone --reference-audio ref.wav --text "..." --output out.wav
//!   cutetts --help
//!
//! Seed: OS-random each run unless --seed N is given; the run always prints
//! `seed <N>` — pass it back to replay bit-identically (same machine/threads).
//! Pipeline: tokenize -> embed lookup -> Qwen prefill -> AR loop
//! (LM decode -> stop? -> DiT 4-step (own xorshift RNG) -> scale ->
//! LocEnc feedback) -> concat latents -> VAE whole decode -> 24k i16 wav.
//! x0 RNG is our own (torch Philox not replicated — chaos makes x0-exactness
//! pointless; trajectory validated bit-exact given same x0 in tests).

use cutetts_codec::conv::default_threads;
use cutetts_codec::decode::decode_nth;
use cutetts_codec::dit::euler_sample;
use cutetts_codec::e2e::{load_all, stop_logits, AllW};
use cutetts_codec::locenc::{locenc_embed, LocencW};
use cutetts_codec::prefix::PrefixW;
use cutetts_codec::qwen::{decode_step, embed_lookup, prefill, QwenCache};
use cutetts_codec::tok::PromptTokenizer;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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

fn help() -> ! {
    print!(
        "\
cutetts (distill, offline): text -> 24k wav, no Python.

usage:
  cutetts --text TEXT --output OUT.wav [options]
  cutetts --text-file FILE --output OUT.wav [options]
  cutetts --mode voice_clone --reference-audio REF.wav --text TEXT --output OUT.wav [options]

options:
  --mode tts|voice_clone   default tts (clone needs --reference-audio)
  --reference-audio PATH   reference wav for voice_clone (any PCM/mono-stereo/rate)
  --text TEXT              text to speak (mutually exclusive with --text-file)
  --text-file PATH         read text from file (trailing newline trimmed;
                         long text auto-splits into sentence chunks, 30ms
                         crossfade join; chunk seeds = base+idx)
  --output PATH            output wav (16-bit mono 24kHz, overwritten)
  --seed N                 u64 seed; default = random each run (printed, reuse to replay)
  --max-steps N            default 750 (each step = 2 latent frames = 0.16s)
  --model-dir DIR          default <manifest>/../model/CuteTTS-distill
  --threads N              override CUTETTS_THREADS / ncpu
  --help, -h               this message

repro: the run prints `seed <N>`; pass it back via --seed for bit-identical output
       on the same machine/thread count. seed 0 is remapped (xorshift guard).
"
    );
    std::process::exit(0);
}

fn usage() -> ! {
    eprintln!("usage: cutetts --text TEXT --output OUT.wav [--mode tts|voice_clone] [--seed N]");
    eprintln!("       cutetts --help for full options");
    std::process::exit(2);
}

fn err(msg: &str) -> ! {
    eprintln!("cutetts: error: {msg}");
    eprintln!("try `cutetts --help`");
    std::process::exit(2);
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn arg_val(args: &[String], name: &str) -> Option<String> {
    // Supports both `--name value` and `--name=value`.
    let eq = format!("{name}=");
    for (i, a) in args.iter().enumerate() {
        if a == name {
            return args.get(i + 1).cloned();
        }
        if let Some(v) = a.strip_prefix(&eq) {
            return Some(v.to_string());
        }
    }
    None
}

/// Random seed when --seed is absent (QORA-style): nanos since epoch,
/// std-only and cross-platform (no /dev/urandom, no rand crate).
/// May be 0 in theory — callers route it through effective_seed.
fn random_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E3779B97F4A7C15)
}

/// xorshift64* gets stuck at state 0 — remap seed 0 to a fixed nonzero.
fn effective_seed(seed: u64) -> u64 {
    if seed == 0 { 0x9E3779B97F4A7C15 } else { seed }
}

/// Read any PCM/float wav to mono f32 + sample rate (hound).
fn read_wav_mono(path: &str) -> (Vec<f32>, usize) {
    let mut reader =
        hound::WavReader::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let spec = reader.spec();
    let ch = spec.channels as usize;
    let frames: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect(),
        (hound::SampleFormat::Int, 24) | (hound::SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .map(|s| (s.unwrap() as f64 / 2147483648.0) as f32)
            .collect(),
        (f, b) => panic!("unsupported wav format {f:?}/{b}: {path}"),
    };
    assert_eq!(frames.len() % ch, 0);
    let mono = cutetts_codec::prefix::to_mono(&frames, ch);
    (mono, spec.sample_rate as usize)
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

/// Precomputed voice_clone reference state (built once, reused per chunk).
struct CloneCtx {
    spk: Vec<f32>,
    feats: Vec<f32>,
    nf: usize,
    lw: LocencW,
    pw: PrefixW,
}

/// Synthesize one text piece -> mono 24k samples. Returns (wav, steps).
/// `chunk_seed` is the effective seed for this piece (caller derives
/// base+idx for multi-chunk runs so replays are deterministic).
fn synth_one(
    w: &AllW,
    tok: &PromptTokenizer,
    text: &str,
    chunk_seed: u64,
    max_steps: usize,
    nth: usize,
    clone: Option<&CloneCtx>,
) -> (Vec<f32>, usize) {
    // Prefix. tts: text only; clone: full reference chain.
    let (prefix, tpre, spk_opt): (Vec<f32>, usize, Option<Vec<f32>>) = match clone {
        None => {
            let ids = tok.encode_tts(text);
            let t = ids.len();
            (embed_lookup(&w.qwen, &ids), t, None)
        }
        Some(cx) => {
            let (embeds, t) = cutetts_codec::prefix::clone_prefix(
                tok, &w.qwen, &cx.lw, &cx.pw, text, &cx.feats, cx.nf, &cx.spk,
                w.e2e.scale, w.e2e.bias, nth,
            );
            (embeds, t, Some(cx.spk.clone()))
        }
    };
    let mut cache = QwenCache::empty();
    let h = prefill(&w.qwen, &prefix, tpre, 0, &mut cache, nth);
    let mut last = h[(tpre - 1) * 1024..tpre * 1024].to_vec();

    let mut rng = Rng(effective_seed(chunk_seed));
    let mut cond = vec![0.0f32; 128]; // initial previous cond = zeros [1,2,64]
    let mut latents: Vec<f32> = Vec::new();
    let mut steps = 0;
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
            let p = euler_sample(&w.dit, &x0, &last, &cond, spk_opt.as_deref(), 4, 2.0, nth);
            let s: Vec<f32> = p.iter().map(|&v| v / w.e2e.scale - w.e2e.bias).collect();
            (p, s)
        };
        // next cond = UNSCALED pred ([2,64] flat, as returned).
        // Feedback MUST be RAW pred too (generation.py:1022 uses pred_latent,
        // not pred_latent_scaled; scaled compounds every step).
        let fb = locenc_embed(&w.locenc, &pred, 1, 1, nth);
        cond = pred;
        latents.extend_from_slice(&scaled);
        last = decode_step(&w.qwen, &fb, tpre + steps, &mut cache);
        steps += 1;
    }

    let nframes = steps * 2;
    let mut frames = vec![0.0f32; 64 * nframes];
    for t in 0..nframes {
        for c in 0..64 {
            frames[c * nframes + t] = latents[t * 64 + c];
        }
    }
    let wav = decode_nth(&w.vae, &frames, nframes, nth);
    (wav, steps)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if has_flag(&args, "--help") || has_flag(&args, "-h") || args.len() < 2 {
        help();
    }
    let mode = arg_val(&args, "--mode").unwrap_or_else(|| "tts".to_string());
    if mode != "tts" && mode != "voice_clone" {
        err(&format!("--mode must be tts or voice_clone, got `{mode}`"));
    }
    let text_opt = arg_val(&args, "--text");
    let text_file_opt = arg_val(&args, "--text-file");
    let from_file = text_file_opt.is_some();
    let text = match (text_opt, text_file_opt) {
        (Some(_), Some(_)) => err("--text and --text-file are mutually exclusive"),
        (None, None) => usage(),
        (Some(t), None) => t,
        (None, Some(f)) => {
            let s = std::fs::read_to_string(&f).unwrap_or_else(|e| panic!("read {f}: {e}"));
            let s = s.trim().to_string();
            if s.is_empty() {
                err(&format!("text file is empty: {f}"));
            }
            s
        }
    };
    let out = arg_val(&args, "--output").unwrap_or_else(|| usage());
    if std::path::Path::new(&out).exists() {
        eprintln!("cutetts: warn: overwriting {out}");
    }
    let ref_opt = arg_val(&args, "--reference-audio");
    if mode == "voice_clone" && ref_opt.is_none() {
        err("--mode voice_clone requires --reference-audio REF.wav");
    }
    if mode == "tts" && ref_opt.is_some() {
        err("--reference-audio needs --mode voice_clone (tts ignores reference)");
    }
    let model_dir = arg_val(&args, "--model-dir").unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../model/CuteTTS-distill")
            .to_string_lossy()
            .into_owned()
    });
    let (seed, seed_src): (u64, &str) = match arg_val(&args, "--seed") {
        Some(v) => (v.parse::<u64>().unwrap_or_else(|_| err("--seed must be a u64")), "user"),
        None => (random_seed(), "random"),
    };
    let max_steps: usize = match arg_val(&args, "--max-steps") {
        Some(v) => v.parse().unwrap_or_else(|_| err("--max-steps must be an integer")),
        None => 750,
    };
    if let Some(v) = arg_val(&args, "--threads") {
        let n: usize = v.parse().unwrap_or_else(|_| err("--threads must be an integer >= 1"));
        if n < 1 {
            err("--threads must be >= 1");
        }
        // Must precede first pool use (pool sizes itself once from this var).
        std::env::set_var("CUTETTS_THREADS", n.to_string());
    }
    println!("cutetts mode={mode} seed={seed} ({seed_src}) max_steps={max_steps}");
    let nth = default_threads();
    println!("threads={nth} model={model_dir}");
    let root = PathBuf::from(&model_dir);
    for p in [
        root.join("tokenizer/tokenizer.model"),
        root.join("weights/tts/model.safetensors"),
        root.join("weights/audio_vae/model.safetensors"),
    ] {
        if !p.is_file() {
            err(&format!("model file missing: {} (--model-dir={model_dir})", p.display()));
        }
    }

    let t0 = Instant::now();
    let tok = PromptTokenizer::load(&root.join("tokenizer/tokenizer.model"));
    let w = load_all(
        &root.join("weights/tts/model.safetensors"),
        &root.join("weights/audio_vae/model.safetensors"),
    );
    println!("weights loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // Reference state once (clone) — reused by every chunk.
    // (mode / ref_opt validated above.)
    let clone_ctx: Option<CloneCtx> = if mode == "tts" {
        None
    } else {
        let ref_path = ref_opt.clone().unwrap();
        let t1 = Instant::now();
        let (mono, sr) = read_wav_mono(&ref_path);
        // torch caps the read at ~30s of source frames
        let cap_frames = (30 * sr + 159) / 160 * 160;
        let mono = if mono.len() > cap_frames { &mono[..cap_frames] } else { &mono[..] };
        let spk_wave = cutetts_codec::prefix::speaker_branch_16k(mono, sr);
        let ref24 = cutetts_codec::prefix::reference_branch_24k(mono, sr);
        let sw = cutetts_codec::speaker::load_speaker_weights(
            &root.join("weights/speaker_encoder/model.safetensors"),
        );
        let ve = cutetts_codec::vae_enc::load_vae_enc_weights(
            &root.join("weights/audio_vae/model.safetensors"),
        );
        let lw = cutetts_codec::locenc::load_locenc_weights(
            &root.join("weights/tts/model.safetensors"),
        );
        let pw = cutetts_codec::prefix::load_prefix_weights(
            &root.join("weights/tts/model.safetensors"),
        );
        let spk = cutetts_codec::speaker::speaker_forward(&sw, &spk_wave, nth);
        let (feats, nf) = cutetts_codec::prefix::reference_features(&ve, &ref24, nth);
        println!(
            "reference: {} -> {} spk frames, {} ref frames ({:.2}s)",
            ref_path,
            spk_wave.len(),
            nf,
            t1.elapsed().as_secs_f32()
        );
        Some(CloneCtx { spk, feats, nf, lw, pw })
    };

    // Long-text chunking (QORA-style): --text-file splits into sentences so
    // each piece stays in-distribution for the small LM (a 1000+ token
    // prefix wanders even in official torch). --text stays single-shot.
    // Chunk seeds = base+idx, deterministic for replay.
    let pieces: Vec<String> = if from_file {
        let v = cutetts_codec::chunk::split_sentences(&text);
        if v.len() > 1 {
            println!("text-file: {} chunks", v.len());
            v
        } else {
            vec![text]
        }
    } else {
        vec![text]
    };
    let t0 = Instant::now();
    let mut wavs: Vec<Vec<f32>> = Vec::with_capacity(pieces.len());
    let mut total_steps = 0;
    for (idx, piece) in pieces.iter().enumerate() {
        let cs = seed.wrapping_add(idx as u64);
        if pieces.len() > 1 {
            println!("--- chunk {}/{} ({} chars, seed={cs}) ---", idx + 1, pieces.len(), piece.chars().count());
        }
        let t1 = Instant::now();
        let (wav, steps) = synth_one(&w, &tok, piece, cs, max_steps, nth, clone_ctx.as_ref());
        println!("chunk {}: {steps} steps, {:.2}s audio ({:.2}s)", idx + 1, wav.len() as f32 / 24000.0, t1.elapsed().as_secs_f32());
        total_steps += steps;
        wavs.push(wav);
    }
    // Seam hygiene (QORA): per-chunk 0.15s tail pad, then 30ms crossfade.
    // Single piece: write as-is (no pad — bit-identical to old behavior).
    let wav: Vec<f32> = if wavs.len() == 1 {
        wavs.pop().unwrap()
    } else {
        let padded: Vec<Vec<f32>> =
            wavs.iter().map(|a| cutetts_codec::chunk::pad_tail(a, 24000, 0.15)).collect();
        cutetts_codec::chunk::crossfade_concat(&padded, 720)
    };
    println!("total: {total_steps} steps in {:.1}s", t0.elapsed().as_secs_f32());
    write_wav_16(PathBuf::from(&out).as_path(), &wav, 24000);
    let dur = wav.len() as f32 / 24000.0;
    println!("wrote {out} ({dur:.2}s audio, seed={seed}, chunks={})", pieces.len());
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

    #[test]
    fn seed_zero_remapped() {
        assert_eq!(effective_seed(0), 0x9E3779B97F4A7C15);
        assert_eq!(effective_seed(42), 42);
        // remapped seed still draws finite normals (no stuck-at-zero state)
        let mut rng = Rng(effective_seed(0));
        assert!(rng.normals(1024).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn arg_val_eq_form() {
        let args = vec!["cutetts".to_string(), "--seed=42".to_string(), "--text".to_string(), "hi".to_string()];
        assert_eq!(arg_val(&args, "--seed"), Some("42".to_string()));
        assert_eq!(arg_val(&args, "--text"), Some("hi".to_string()));
        assert_eq!(arg_val(&args, "--missing"), None);
    }

    #[test]
    fn random_seed_nonzero() {
        // Statistical, not strict: just guards a broken clock path.
        for _ in 0..10 {
            if random_seed() != 0 {
                return;
            }
        }
        panic!("random_seed stuck at 0");
    }
}
