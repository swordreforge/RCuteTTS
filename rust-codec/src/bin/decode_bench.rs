//! Native Rust decode benchmark (no FFI, no Python).
//!
//! Usage:
//!   cutetts-decode <weights.safetensors> <latent.npy> [--ref ref.npy] [--repeat N] [--threads N] [--out out.npy]
//!
//! `latent.npy`: f32 `[64, frames]` channel-first (same as `testdata/m2_lat_*.npy`).
//! Prints one line per repeat + a summary with RTF.

use cutetts_codec::conv::default_threads;
use cutetts_codec::decode::decode_nth;
use cutetts_codec::decoder::HOP_LENGTH;
use cutetts_codec::weights::load_decoder_weights;
use std::path::PathBuf;
use std::time::Instant;

fn usage() -> ! {
    eprintln!(
        "usage: cutetts-decode <weights.safetensors> <latent.npy> [--ref ref.npy] [--repeat N] [--threads N] [--out out.npy]"
    );
    std::process::exit(2);
}

fn main() {
    let mut args = std::env::args().skip(1);
    let weights = PathBuf::from(args.next().unwrap_or_else(|| usage()));
    let latent_path = PathBuf::from(args.next().unwrap_or_else(|| usage()));
    let mut ref_path: Option<PathBuf> = None;
    let mut repeat = 3usize;
    let mut threads = default_threads();
    let mut out_path: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--ref" => ref_path = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            "--repeat" => repeat = args.next().unwrap_or_else(|| usage()).parse().unwrap_or_else(|_| usage()),
            "--threads" => threads = args.next().unwrap_or_else(|| usage()).parse().unwrap_or_else(|_| usage()),
            "--out" => out_path = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            _ => usage(),
        }
    }

    let t0 = Instant::now();
    let w = load_decoder_weights(&weights);
    let load_s = t0.elapsed().as_secs_f32();

    let lat: ndarray::Array2<f32> = ndarray_npy::read_npy(&latent_path).unwrap_or_else(|e| {
        eprintln!("read latent {:?}: {e}", latent_path);
        std::process::exit(1);
    });
    assert_eq!(lat.shape()[0], 64, "latent must be [64, frames], got {:?}", lat.shape());
    let frames = lat.shape()[1];
    let latent = lat.as_slice().expect("latent contiguous").to_vec();
    let audio_s = frames as f32 * HOP_LENGTH as f32 / 24000.0;

    // warmup (compile caches, thread pool, page faults out of the timed region)
    let wav0 = decode_nth(&w, &latent, frames, threads);
    assert_eq!(wav0.len(), frames * HOP_LENGTH);

    let mut times = Vec::with_capacity(repeat);
    let mut wav = wav0;
    for i in 0..repeat {
        let t1 = Instant::now();
        wav = decode_nth(&w, &latent, frames, threads);
        let dt = t1.elapsed().as_secs_f32();
        times.push(dt);
        println!("repeat[{i}]: {dt:.3}s rtf={:.3}", dt / audio_s);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    println!(
        "summary: frames={frames} audio={audio_s:.2}s threads={threads} load={load_s:.2}s median={median:.3}s rtf={:.3}",
        median / audio_s
    );

    if let Some(rp) = ref_path {
        let refr: ndarray::Array2<f32> = ndarray_npy::read_npy(&rp).unwrap_or_else(|e| {
            eprintln!("read ref {:?}: {e}", rp);
            std::process::exit(1);
        });
        let r = refr.as_slice().expect("ref contiguous");
        assert_eq!(r.len(), wav.len(), "ref len {} vs wav len {}", r.len(), wav.len());
        let err = wav.iter().zip(r.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        println!("max_abs_err={err:.2e}");
    }

    if let Some(op) = out_path {
        let arr = ndarray::Array2::from_shape_vec((1, wav.len()), wav).unwrap();
        ndarray_npy::write_npy(&op, &arr).unwrap();
        println!("wrote {}", op.display());
    }
}
