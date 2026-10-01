//! Full offline BASE voice_clone ring vs torch
//! (`scripts/dump_e2e_base_clone.py`, 60 steps, default_reference.wav).
//!
//! Same chaotic-loop rules as the other e2e tests: no waveform point-parity
//! on free runs. Clone-specific chain (ECAPA/resample/VAE-enc/prefix/speaker
//! linear) is variant-independent and already gated under distill; the base
//! delta is the DiT speaker branch (cond adaln(spk), uncond adaln(zeros)),
//! gated per-step here plus the stop sequence on a true dual-branch ring.
use cutetts_codec::decode::decode_nth;
use cutetts_codec::e2e::{dit_step_cfg, load_all, stop_logits};
use cutetts_codec::locenc::locenc_embed;
use cutetts_codec::qwen::{decode_step, prefill, QwenCache};
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

const STEPS_CFG: usize = 10;
const SWAY: f32 = -0.8;
const CFG: f32 = 2.0;

fn load() -> Option<(cutetts_codec::e2e::AllW, usize, usize, usize, Vec<bool>, Vec<f32>)> {
    let tts = manifest().join("../model/CuteTTS/weights/tts/model.safetensors");
    let vae = manifest().join("../model/CuteTTS/weights/audio_vae/model.safetensors");
    if !tts.is_file() || !vae.is_file() {
        eprintln!("skip: weights missing");
        return None;
    }
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(manifest().join("e2ebc_meta.json")).unwrap()).unwrap();
    let w = load_all(&tts, &vae);
    let tpre = load_flat("e2ebc_prefix").len() / 1024;
    let tupre = load_flat("e2ebc_uprefix").len() / 1024;
    let steps = meta["steps"].as_u64().unwrap() as usize;
    let stops: Vec<bool> = serde_json::from_value(meta["stops"].clone()).unwrap();
    let spk = load_flat("e2ebc_spkemb");
    assert_eq!(spk.len(), 256);
    assert_eq!(tupre, 1, "base uncond prefix is 1 token");
    Some((w, tpre, tupre, steps, stops, spk))
}

#[test]
fn e2e_base_clone_teacher_forced() {
    let (w, _tpre, _tupre, steps, stops, spk) = match load() {
        Some(v) => v,
        None => return,
    };
    let nth = cutetts_codec::conv::default_threads();
    let mut latents: Vec<f32> = Vec::with_capacity(steps * 128);
    let mut worst_pred = 0.0f32;
    let mut worst_stop = 0;
    for i in 0..steps {
        let x0 = load_flat(&format!("e2ebc_{i:02}_x0"));
        let zc = load_flat(&format!("e2ebc_{i:02}_zc"));
        let zu = load_flat(&format!("e2ebc_{i:02}_zu"));
        let cond = load_flat(&format!("e2ebc_{i:02}_cond"));
        let uncond = load_flat(&format!("e2ebc_{i:02}_uncond"));
        let pref = load_flat(&format!("e2ebc_{i:02}_pred"));
        let (pred, scaled) =
            dit_step_cfg(&w, &x0, &zc, &zu, &cond, &uncond, Some(&spk), STEPS_CFG, SWAY, CFG, nth);
        worst_pred = worst_pred.max(max_err(&pred, &pref));
        let sl = stop_logits(&w.e2e, &zc);
        if (sl[1] > sl[0]) != stops[i] {
            worst_stop += 1;
        }
        latents.extend_from_slice(&scaled);
    }
    println!("base-clone teacher-forced DiT-CFG worst={worst_pred:.2e} stop_mismatch={worst_stop}");
    assert!(worst_pred < 5e-4, "dit-cfg {worst_pred:.2e}");
    assert_eq!(worst_stop, 0);

    let nframes = steps * 2;
    let mut frames = vec![0.0f32; 64 * nframes];
    for t in 0..nframes {
        for c in 0..64 {
            frames[c * nframes + t] = latents[t * 64 + c];
        }
    }
    let wav = decode_nth(&w.vae, &frames, nframes, nth);
    let refwav = load_flat("e2ebc_wav");
    assert_eq!(wav.len(), refwav.len());
    let ew = max_err(&wav, &refwav);
    println!("base-clone teacher-forced wav err={ew:.2e}");
    assert!(ew < 5e-4, "wav {ew:.2e}");
    println!("BASE-CLONE TEACHER-FORCED PASSED");
}

#[test]
fn e2e_base_clone_true_ring() {
    let (w, tpre, tupre, steps, stops, spk) = match load() {
        Some(v) => v,
        None => return,
    };
    let nth = cutetts_codec::conv::default_threads();
    let prefix = load_flat("e2ebc_prefix");
    let uprefix = load_flat("e2ebc_uprefix");
    let mut cache = QwenCache::empty();
    let h = prefill(&w.qwen, &prefix, tpre, 0, &mut cache, nth);
    let mut last = h[(tpre - 1) * 1024..tpre * 1024].to_vec();
    let mut ucache = QwenCache::empty();
    let uh = prefill(&w.qwen, &uprefix, tupre, 0, &mut ucache, nth);
    let mut ulast = uh[(tupre - 1) * 1024..tupre * 1024].to_vec();
    let mut upos = tupre;
    let mut cond = vec![0.0f32; 128];
    let mut uncond = vec![0.0f32; 128];
    let mut mismatch = 0;
    let mut worst_loc = 0.0f32;
    for i in 0..steps {
        let x0 = load_flat(&format!("e2ebc_{i:02}_x0"));
        let (pred, _) = dit_step_cfg(
            &w, &x0, &last, &ulast, &cond, &uncond, Some(&spk), STEPS_CFG, SWAY, CFG, nth,
        );
        let sl = stop_logits(&w.e2e, &last);
        if (sl[1] > sl[0]) != stops[i] {
            mismatch += 1;
        }
        if i + 1 < steps {
            let fb = locenc_embed(&w.locenc, &pred, 1, 1, nth);
            let lmin = load_flat(&format!("e2ebc_{i:02}_lmin"));
            worst_loc = worst_loc.max(max_err(&fb, &lmin));
            last = decode_step(&w.qwen, &fb, tpre + i, &mut cache);
            ulast = decode_step(&w.qwen, &fb, upos, &mut ucache);
            upos += 1;
            cond = pred.clone();
            uncond = pred;
        }
    }
    println!("base-clone true-ring: loc={worst_loc:.2e} stop_mismatch={mismatch}");
    assert!(worst_loc < 1.0, "feedback loc drift {worst_loc:.2e}");
    assert_eq!(mismatch, 0, "stop sequence must match exactly");
    println!("BASE-CLONE TRUE-RING stops match");
}
