//! Streaming decode parity: M2 vectors in small chunks vs whole-sentence.
//! Gate 1e-6 (same taps => expected bit-identical). Also reports first-chunk
//! latency and runs the e2e teacher-forced trajectory through the streamer
//! (AR-loop context: state carried across 13 real patches).
use cutetts_codec::decode::decode_nth;
use cutetts_codec::stream::StreamingDecoder;
use cutetts_codec::weights::load_decoder_weights;
use ndarray_npy::read_npy;
use std::path::PathBuf;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn load2(name: &str) -> Vec<f32> {
    let p = manifest().join(format!("testdata/{name}.npy"));
    if let Ok(a) = read_npy::<_, ndarray::Array2<f32>>(p.clone()) {
        return a.into_raw_vec_and_offset().0;
    }
    read_npy::<_, ndarray::Array3<f32>>(p).unwrap().into_raw_vec_and_offset().0
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn wpath() -> PathBuf {
    manifest().join("../model/CuteTTS/weights/audio_vae/model.safetensors")
}

#[test]
fn stream_chunks_match_offline() {
    if !wpath().is_file() {
        eprintln!("skip: weights missing");
        return;
    }
    let w = load_decoder_weights(&wpath());
    let nth = cutetts_codec::conv::default_threads();
    let mut worst = 0.0f32;
    // m2 vectors have frames 2..24; chunk into 2-frame pieces (AR patch size)
    for i in 0..10 {
        let lat = load2(&format!("m2_lat_{i:02}"));
        let refr = load2(&format!("m2_ref_{i:02}"));
        let frames = lat.len() / 64;
        assert_eq!(frames * 64, lat.len());
        let mut dec = StreamingDecoder::new(&w, nth);
        let t0 = std::time::Instant::now();
        let mut wav = Vec::new();
        let mut first_ms = 0.0f32;
        let mut f = 0;
        while f < frames {
            let n = (2).min(frames - f);
            let chunk: Vec<f32> = (0..64).flat_map(|c| lat[c * frames + f..c * frames + f + n].to_vec()).collect();
            let t1 = std::time::Instant::now();
            let out = dec.decode_chunk(&chunk, n);
            if f == 0 {
                first_ms = t1.elapsed().as_secs_f32() * 1000.0;
            }
            wav.extend_from_slice(&out);
            f += n;
        }
        let e = max_err(&wav, &refr);
        worst = worst.max(e);
        println!("m2_{i:02}: frames={frames} err={e:.2e} first-chunk={first_ms:.1}ms");
        assert!(e < 1e-6, "m2_{i:02} {e:.2e}");
    }
    println!("STREAM gate passed, worst={worst:.2e}");
}

#[test]
fn stream_odd_chunks_match_offline() {
    // Odd chunk sizes (3,5,7) — history bookkeeping must hold for any split.
    if !wpath().is_file() {
        eprintln!("skip: weights missing");
        return;
    }
    let w = load_decoder_weights(&wpath());
    let nth = cutetts_codec::conv::default_threads();
    let lat = load2("m2_lat_09"); // 24 frames
    let refr = load2("m2_ref_09");
    for &cs in &[3usize, 5, 7] {
        let mut dec = StreamingDecoder::new(&w, nth);
        let mut wav = Vec::new();
        let mut f = 0;
        while f < 24 {
            let n = cs.min(24 - f);
            let chunk: Vec<f32> = (0..64).flat_map(|c| lat[c * 24 + f..c * 24 + f + n].to_vec()).collect();
            wav.extend_from_slice(&dec.decode_chunk(&chunk, n));
            f += n;
        }
        let e = max_err(&wav, &refr);
        println!("chunksize {cs}: err={e:.2e}");
        assert!(e < 1e-6, "cs={cs} {e:.2e}");
    }
}

#[test]
fn stream_e2e_trajectory() {
    // Real AR-loop context: 13 teacher-forced patches back-to-back.
    if !wpath().is_file() {
        eprintln!("skip: weights missing");
        return;
    }
    let tts = manifest().join("../model/CuteTTS-distill/weights/tts/model.safetensors");
    if !tts.is_file() {
        eprintln!("skip: tts weights missing");
        return;
    }
    let w = load_decoder_weights(&wpath());
    let nth = cutetts_codec::conv::default_threads();
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(manifest().join("e2e_meta.json")).unwrap(),
    )
    .unwrap();
    let steps = meta["steps"].as_u64().unwrap() as usize;
    // teacher-forced scaled latents: rebuild from e2e test? They aren't saved;
    // re-derive: scaled = pred/scale - bias using saved preds.
    let scale = meta["scale"].as_f64().unwrap() as f32;
    let bias = meta["bias"].as_f64().unwrap() as f32;
    let mut dec = StreamingDecoder::new(&w, nth);
    let mut wav = Vec::new();
    let t0 = std::time::Instant::now();
    for i in 0..steps {
        let pred = load2(&format!("e2e_{i:02}_pred")); // [2,64] rows
        // [T=2,C=64] rows -> [64,2] channel-first chunk
        let mut chunk = vec![0.0f32; 128];
        for f in 0..2 {
            for c in 0..64 {
                chunk[c * 2 + f] = pred[f * 64 + c] / scale - bias;
            }
        }
        let out = dec.decode_chunk(&chunk, 2);
        if i == 0 {
            println!("first patch decode: {:.1}ms", t0.elapsed().as_secs_f32() * 1000.0);
        }
        wav.extend_from_slice(&out);
    }
    let refwav = load2("e2e_wav");
    assert_eq!(wav.len(), refwav.len());
    let e = max_err(&wav, &refwav);
    println!("stream e2e trajectory (teacher-forced preds): err={e:.2e}");
    // Pinned trajectory: must match offline closely (offline wav err was 2.2e-5).
    assert!(e < 1e-3, "stream e2e {e:.2e}");
}

#[test]
fn stream_offline_parity() {
    // Streaming the whole utterance at once must equal offline decode.
    if !wpath().is_file() {
        eprintln!("skip: weights missing");
        return;
    }
    let w = load_decoder_weights(&wpath());
    let nth = cutetts_codec::conv::default_threads();
    let lat = load2("m2_lat_09");
    let refr = load2("m2_ref_09");
    let off = decode_nth(&w, &lat, 24, nth);
    let mut dec = StreamingDecoder::new(&w, nth);
    let got = dec.decode_chunk(&lat, 24);
    let e = max_err(&got, &off);
    let e2 = max_err(&got, &refr);
    println!("single-chunk vs offline: {e:.2e}, vs torch: {e2:.2e}");
    assert!(e < 1e-6);
}
