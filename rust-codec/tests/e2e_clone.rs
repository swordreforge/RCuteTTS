//! Voice-clone ring vs torch (`scripts/dump_clone_vectors.py`, 18 steps).
//! Same strategy as e2e_tts: teacher-forced strict + true-ring stop gates.
//! DiT runs WITH adaln (fixture speaker); ECAPA itself is gated in
//! tests/speaker.rs (emb 2.2e-5, fp32 both sides).
use cutetts_codec::decode::decode_nth;
use cutetts_codec::e2e::{dit_step, load_all, stop_logits};
use cutetts_codec::locenc::locenc_embed;
use cutetts_codec::qwen::{decode_step, prefill, QwenCache};
use cutetts_codec::speaker::{load_speaker_weights, speaker_forward};
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
    if let Ok(a) = read_npy::<_, ndarray::Array2<f32>>(p.clone()) {
        return a.into_raw_vec_and_offset().0;
    }
    read_npy::<_, ndarray::Array1<f32>>(p).unwrap().to_vec()
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn load_ring() -> Option<(cutetts_codec::e2e::AllW, Vec<f32>, usize, usize, Vec<bool>, usize)> {
    let tts = manifest().join("../model/CuteTTS-distill/weights/tts/model.safetensors");
    let vae = manifest().join("../model/CuteTTS-distill/weights/audio_vae/model.safetensors");
    let spk = manifest().join("../model/CuteTTS-distill/weights/speaker_encoder/model.safetensors");
    if !tts.is_file() || !vae.is_file() || !spk.is_file() {
        eprintln!("skip: weights missing");
        return None;
    }
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(manifest().join("clone_meta.json")).unwrap(),
    )
    .unwrap();
    let nth = cutetts_codec::conv::default_threads();
    let t0 = std::time::Instant::now();
    let w = load_all(&tts, &vae);
    let sw = load_speaker_weights(&spk);
    println!("weights loaded in {:.1}s", t0.elapsed().as_secs_f32());
    // ECAPA on the fixture wave (also asserted separately in speaker test)
    let wave = load_flat("spk_wave");
    let t1 = std::time::Instant::now();
    let emb = speaker_forward(&sw, &wave, nth);
    println!("ecapa: {:.2}s err={:.2e}", t1.elapsed().as_secs_f32(), max_err(&emb, &load_flat("spk_emb")));
    let prefix = load_flat("clone_prefix");
    let tpre = prefix.len() / 1024;
    Some((
        w,
        prefix,
        tpre,
        meta["steps"].as_u64().unwrap() as usize,
        serde_json::from_value(meta["stops"].clone()).unwrap(),
        nth,
    ))
}

#[test]
fn clone_teacher_forced() {
    let (w, prefix, tpre, steps, stops, nth) = match load_ring() {
        Some(r) => r,
        None => return,
    };
    let spk = load_flat("spk_emb");
    let mut cache = QwenCache::empty();
    let _ = prefill(&w.qwen, &prefix, tpre, 0, &mut cache, nth);
    let mut latents: Vec<f32> = Vec::with_capacity(steps * 128);
    let mut worst_pred = 0.0f32;
    let mut worst_stop = 0;
    for i in 0..steps {
        let x0 = load_flat(&format!("clone_{i:02}_x0"));
        let z = load_flat(&format!("clone_{i:02}_z"));
        let cond = load_flat(&format!("clone_{i:02}_cond"));
        let pref = load_flat(&format!("clone_{i:02}_pred"));
        let (pred, scaled) = dit_step(&w, &x0, &z, &cond, Some(&spk), nth);
        worst_pred = worst_pred.max(max_err(&pred, &pref));
        let sl = stop_logits(&w.e2e, &z);
        if (sl[1] > sl[0]) != stops[i] {
            worst_stop += 1;
        }
        latents.extend_from_slice(&scaled);
    }
    println!("clone teacher-forced DiT worst={worst_pred:.2e} stop_mismatch={worst_stop}");
    assert!(worst_pred < 1e-4, "dit {worst_pred:.2e}");
    assert_eq!(worst_stop, 0);
    let nframes = steps * 2;
    let mut frames = vec![0.0f32; 64 * nframes];
    for t in 0..nframes {
        for c in 0..64 {
            frames[c * nframes + t] = latents[t * 64 + c];
        }
    }
    let wav = decode_nth(&w.vae, &frames, nframes, nth);
    let refwav = load_flat("clone_wav");
    assert_eq!(wav.len(), refwav.len());
    let ew = max_err(&wav, &refwav);
    println!("clone teacher-forced wav err={ew:.2e}");
    assert!(ew < 1e-3, "wav {ew:.2e}");
    println!("CLONE TEACHER-FORCED PASSED");
}

#[test]
fn clone_true_ring_drift() {
    let (w, prefix, tpre, steps, stops, nth) = match load_ring() {
        Some(r) => r,
        None => return,
    };
    let wave = load_flat("spk_wave");
    let sw = load_speaker_weights(
        &manifest().join("../model/CuteTTS-distill/weights/speaker_encoder/model.safetensors"),
    );
    let spk = speaker_forward(&sw, &wave, nth);
    let mut cache = QwenCache::empty();
    let h = prefill(&w.qwen, &prefix, tpre, 0, &mut cache, nth);
    let mut last = h[(tpre - 1) * 1024..tpre * 1024].to_vec();
    let mut mismatch = 0;
    let mut worst_dz = 0.0f32;
    for i in 0..steps {
        let x0 = load_flat(&format!("clone_{i:02}_x0"));
        let zref = load_flat(&format!("clone_{i:02}_z"));
        let condref = load_flat(&format!("clone_{i:02}_cond"));
        let dz = max_err(&last, &zref);
        worst_dz = worst_dz.max(dz);
        let (pred, _) = dit_step(&w, &x0, &last, &condref, Some(&spk), nth);
        let sl = stop_logits(&w.e2e, &last);
        let stop = sl[1] > sl[0];
        if stop != stops[i] {
            mismatch += 1;
            println!("  step {i:02}: STOP MISMATCH (ours={stop} torch={}) drift-z={dz:.2e} logits=[{:.2},{:.2}]", stops[i], sl[0], sl[1]);
            // Hard gate only in the low-drift regime; beyond drift-z 3 the
            // loop is chaotic (torch self-noise flips stops AND length 10x).
            assert!(dz > 3.0, "stop mismatch at low drift step {i} (dz={dz:.2e})");
        }
        if i + 1 < steps {
            // Feedback is RAW pred (generation.py:1022), not scaled.
            let fb = locenc_embed(&w.locenc, &pred, 1, 1, nth);
            last = decode_step(&w.qwen, &fb, tpre + i, &mut cache);
        }
    }
    println!("clone true-ring: worst drift-z={worst_dz:.2e} stop_mismatch={mismatch} (all in chaos regime)");
    println!("CLONE TRUE-RING done");
}
