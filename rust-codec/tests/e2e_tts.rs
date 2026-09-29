//! Full offline tts ring vs torch (`scripts/dump_e2e_tts.py`, 13 steps).
//!
//! The AR loop is CHAOTIC: torch itself diverges 0.615 (and changes length!)
//! under σ=0.05 prefix noise, so sample-wise wav parity is unachievable AND
//! meaningless — even torch-vs-torch. Validation strategy:
//! - teacher-forced ring (fixture z per step): DiT pinned to torch's
//!   trajectory; gates strict (pred ~1e-5, wav ~1e-3). THIS is the real gate.
//! - true ring (own state): documents fp32-vs-bf16 drift honestly; only the
//!   stop sequence is gated (it matches 13/13 — linguistic trajectory kept).
use cutetts_codec::decode::decode_nth;
use cutetts_codec::e2e::{dit_step, load_all, stop_logits, AllW};
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

struct Ring {
    w: AllW,
    nth: usize,
    tpre: usize,
    steps: usize,
    stops: Vec<bool>,
}

impl Ring {
    fn load() -> Option<Self> {
        let tts = manifest().join("../model/CuteTTS-distill/weights/tts/model.safetensors");
        let vae = manifest().join("../model/CuteTTS-distill/weights/audio_vae/model.safetensors");
        if !tts.is_file() || !vae.is_file() {
            eprintln!("skip: weights missing");
            return None;
        }
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(manifest().join("e2e_meta.json")).unwrap(),
        )
        .unwrap();
        let t0 = std::time::Instant::now();
        let w = load_all(&tts, &vae);
        println!("weights loaded in {:.1}s", t0.elapsed().as_secs_f32());
        let prefix = load_flat("e2e_prefix");
        Some(Ring {
            w,
            nth: cutetts_codec::conv::default_threads(),
            tpre: prefix.len() / 1024,
            steps: meta["steps"].as_u64().unwrap() as usize,
            stops: serde_json::from_value(meta["stops"].clone()).unwrap(),
        })
    }

    fn run_prefill(&self, cache: &mut Vec<QwenCache>) -> Vec<f32> {
        let prefix = load_flat("e2e_prefix");
        let h = prefill(&self.w.qwen, &prefix, self.tpre, 0, cache, self.nth);
        h[(self.tpre - 1) * 1024..self.tpre * 1024].to_vec()
    }
}

#[test]
fn e2e_teacher_forced() {
    let r = match Ring::load() {
        Some(r) => r,
        None => return,
    };
    let mut cache = QwenCache::empty();
    let _ = r.run_prefill(&mut cache);    let mut latents: Vec<f32> = Vec::with_capacity(r.steps * 128);
    let mut worst_pred = 0.0f32;
    let mut worst_stop = 0;
    for i in 0..r.steps {
        let x0 = load_flat(&format!("e2e_{i:02}_x0"));
        let z = load_flat(&format!("e2e_{i:02}_z"));
        let cond = load_flat(&format!("e2e_{i:02}_cond"));
        let pref = load_flat(&format!("e2e_{i:02}_pred"));
        let (pred, scaled) = dit_step(&r.w, &x0, &z, &cond, None, r.nth);
        let ed = max_err(&pred, &pref);
        worst_pred = worst_pred.max(ed);
        let sl = stop_logits(&r.w.e2e, &z);
        if (sl[1] > sl[0]) != r.stops[i] {
            worst_stop += 1;
        }
        latents.extend_from_slice(&scaled);
    }
    println!("teacher-forced DiT worst={worst_pred:.2e} stop_mismatch={worst_stop}");
    assert!(worst_pred < 1e-4, "dit {worst_pred:.2e}");
    assert_eq!(worst_stop, 0);

    let nframes = r.steps * 2;
    // latents are [T=26, C=64] row-major; VAE wants channel-first [64, T].
    let mut frames = vec![0.0f32; 64 * nframes];
    for t in 0..nframes {
        for c in 0..64 {
            frames[c * nframes + t] = latents[t * 64 + c];
        }
    }
    let wav = decode_nth(&r.w.vae, &frames, nframes, r.nth);
    let refwav = load_flat("e2e_wav");
    assert_eq!(wav.len(), refwav.len());
    let ew = max_err(&wav, &refwav);
    println!("teacher-forced wav err={ew:.2e}");
    assert!(ew < 3e-4, "wav {ew:.2e}");
    println!("TEACHER-FORCED PASSED");
}

#[test]
fn e2e_true_ring_drift() {
    // Honest drift documentation. Only stop sequence is gated.
    let r = match Ring::load() {
        Some(r) => r,
        None => return,
    };
    let mut cache = QwenCache::empty();
    let mut last = r.run_prefill(&mut cache);
    let mut mismatch = 0;
    let mut worst_dz = 0.0f32;
    for i in 0..r.steps {
        let x0 = load_flat(&format!("e2e_{i:02}_x0"));
        let zref = load_flat(&format!("e2e_{i:02}_z"));
        let condref = load_flat(&format!("e2e_{i:02}_cond"));
        worst_dz = worst_dz.max(max_err(&last, &zref));
        let (_, scaled) = dit_step(&r.w, &x0, &last, &condref, None, r.nth);
        let sl = stop_logits(&r.w.e2e, &last);
        if (sl[1] > sl[0]) != r.stops[i] {
            mismatch += 1;
        }
        if i + 1 < r.steps {
            let fb = locenc_embed(&r.w.locenc, &scaled, 1, 1, r.nth);
            last = decode_step(&r.w.qwen, &fb, r.tpre + i, &mut cache);
        }
    }
    println!("true-ring: worst drift-z={worst_dz:.2e} stop_mismatch={mismatch}");
    assert_eq!(mismatch, 0, "stop sequence must match exactly");
    println!("TRUE-RING stop sequence 13/13 match");
}
