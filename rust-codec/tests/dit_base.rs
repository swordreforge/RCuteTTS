//! Base DiT differential tests vs torch (`scripts/dump_base_vectors.py`).
//! Base: no step_size/cfg_strength embeddings, sway Euler grid (10 steps,
//! coeff -0.8), dual-branch CFG v = vc + 2.0*(vc - vu). Gates per PLAN §6:
//! grid 1e-6, predict/euler at the fp32 noise floor (distill gates 1e-4).
use cutetts_codec::dit::{euler_sample_cfg, load_dit_weights, predict, sway_timesteps};
use cutetts_codec::tok::PromptTokenizer;
use ndarray_npy::read_npy;
use std::path::PathBuf;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn base_weights() -> Option<PathBuf> {
    let wpath = manifest().join("../model/CuteTTS/weights/tts/model.safetensors");
    if !wpath.is_file() {
        eprintln!("skip: weights not found at {}", wpath.display());
        return None;
    }
    Some(wpath)
}

fn load_vec(name: &str) -> Vec<f32> {
    let p = manifest().join(format!("testdata/{name}.npy"));
    if let Ok(a) = read_npy::<_, ndarray::Array1<f32>>(p.clone()) {
        return a.to_vec();
    }
    if let Ok(a) = read_npy::<_, ndarray::Array2<f32>>(p.clone()) {
        return a.into_raw_vec_and_offset().0;
    }
    let a = read_npy::<_, ndarray::Array3<f32>>(p).unwrap();
    a.into_raw_vec_and_offset().0
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn base_sway_grid() {
    let refr = load_vec("base_sway");
    assert_eq!(refr.len(), 11);
    let got = sway_timesteps(10, -0.8);
    let e = max_err(&got, &refr);
    println!("sway grid err={e:.2e} t0={:.4} t10={:.4}", got[0], got[10]);
    assert!((got[0] - 0.0).abs() < 1e-6);
    assert!((got[10] - 1.0).abs() < 1e-6);
    assert!(e < 1e-6, "sway grid {e:.2e}");
}

#[test]
fn base_predict_no_embeddings() {
    let wpath = match base_weights() {
        Some(p) => p,
        None => return,
    };
    let w = load_dit_weights(&wpath);
    assert!(w.step_mlp1.is_none() && w.cfg_emb0.is_none(), "base has no embeddings");
    for i in 0..2 {
        let x = load_vec(&format!("base_p{i:02}_x"));
        let t = load_vec(&format!("base_p{i:02}_t"));
        let z = load_vec(&format!("base_p{i:02}_z"));
        let cond = load_vec(&format!("base_p{i:02}_cond"));
        let refr = load_vec(&format!("base_p{i:02}_out"));
        let v = predict(&w, &x, t[0], &z, &cond, 0.0, None, 0.0, 1);
        let e = max_err(&v, &refr);
        println!("base_p{i:02}: err={e:.2e}");
        assert!(e < 1e-4, "base_p{i:02} {e:.2e}");
    }
    println!("BASE predict gate passed");
}

#[test]
fn base_euler_cfg_replay() {
    let wpath = match base_weights() {
        Some(p) => p,
        None => return,
    };
    let w = load_dit_weights(&wpath);
    let nth = cutetts_codec::conv::default_threads();
    for i in 0..2 {
        let zc = load_vec(&format!("base_s{i:02}_zc"));
        let zu = load_vec(&format!("base_s{i:02}_zu"));
        let cond = load_vec(&format!("base_s{i:02}_cond"));
        let uncond = load_vec(&format!("base_s{i:02}_uncond"));
        let x0 = load_vec(&format!("base_s{i:02}_x0"));
        let refr = load_vec(&format!("base_s{i:02}_out"));
        let t0 = std::time::Instant::now();
        let got = euler_sample_cfg(&w, &x0, &zc, &zu, &cond, &uncond, None, 10, -0.8, 2.0, nth);
        let e = max_err(&got, &refr);
        println!("base_s{i:02}: err={e:.2e} ({:.2}s)", t0.elapsed().as_secs_f32());
        assert!(e < 5e-4, "base_s{i:02} {e:.2e}");
    }
    println!("BASE euler-CFG gate passed");
}

#[test]
fn base_predict_speaker_adaln() {
    // clone branch: cond row runs adaln(spk); validates the adaln path
    // against torch with a real-shaped [256] embedding.
    let wpath = match base_weights() {
        Some(p) => p,
        None => return,
    };
    let w = load_dit_weights(&wpath);
    let x = load_vec("base_spk_x");
    let z = load_vec("base_spk_z");
    let cond = load_vec("base_spk_cond");
    let spk = load_vec("base_spk_spk");
    let refr = load_vec("base_spk_out");
    assert_eq!((x.len(), z.len(), cond.len(), spk.len(), refr.len()), (128, 1024, 128, 256, 128));
    let v = predict(&w, &x, 0.3, &z, &cond, 0.0, Some(&spk), 0.0, 1);
    let e = max_err(&v, &refr);
    println!("base_spk: err={e:.2e}");
    assert!(e < 1e-4, "base_spk {e:.2e}");
    println!("BASE speaker-adaln gate passed");
}

#[test]
fn base_euler_cfg_speaker_branches() {
    // full 10-step sway euler: cond branch adaln(spk), uncond branch
    // adaln(zeros) computed explicitly (torch feeds the zeros row through
    // the bias-less speaker_adaln, NOT the plain path).
    let wpath = match base_weights() {
        Some(p) => p,
        None => return,
    };
    let w = load_dit_weights(&wpath);
    let nth = cutetts_codec::conv::default_threads();
    let zc = load_vec("base_ss_zc");
    let zu = load_vec("base_ss_zu");
    let cond = load_vec("base_ss_cond");
    let uncond = load_vec("base_ss_uncond");
    let spk = load_vec("base_ss_spk");
    let x0 = load_vec("base_ss_x0");
    let refr = load_vec("base_ss_out");
    let t0 = std::time::Instant::now();
    let got = euler_sample_cfg(&w, &x0, &zc, &zu, &cond, &uncond, Some(&spk), 10, -0.8, 2.0, nth);
    let e = max_err(&got, &refr);
    println!("base_ss: err={e:.2e} ({:.2}s)", t0.elapsed().as_secs_f32());
    assert!(e < 5e-4, "base_ss {e:.2e}");
    println!("BASE euler-CFG-speaker gate passed");
}

#[test]
fn base_prefixes_exact() {
    // cond = full tts prompt (text rows bitwise), uncond = suffix token only.
    let wpath = match base_weights() {
        Some(p) => p,
        None => return,
    };
    let tok = PromptTokenizer::load(
        &manifest().join("../model/CuteTTS/tokenizer/tokenizer.model"),
    );
    let qw = cutetts_codec::qwen::load_qwen_weights(&wpath);
    let ids = tok.encode_tts("Hello world, this is a test.");
    let cond = cutetts_codec::qwen::embed_lookup(&qw, &ids);
    let refr_c = load_vec("base_cond_prefix");
    assert_eq!(cond.len(), refr_c.len(), "cond len {} vs {}", cond.len(), refr_c.len());
    let ec = max_err(&cond, &refr_c);
    println!("base cond prefix err={ec:.2e} (T={})", ids.len());
    assert!(ec == 0.0, "cond text rows must be bitwise");
    // uncond: single suffix token; id must match torch's suffix encoding
    let suids = tok.encode("<|endofprompt|>");
    assert_eq!(suids.len(), 1, "suffix is 1 token: {suids:?}");
    let u = cutetts_codec::qwen::embed_lookup(&qw, &suids);
    let refr_u = load_vec("base_uncond_prefix");
    assert_eq!(u.len(), refr_u.len());
    let eu = max_err(&u, &refr_u);
    println!("base uncond prefix err={eu:.2e}");
    assert!(eu == 0.0, "uncond row must be bitwise");
    println!("BASE prefix gate passed");
}

#[test]
fn base_euler_fused_bitwise() {
    // CFG panel fusion gate: fused must be BITWISE equal to unfused
    // (same kernels, same per-branch op order — shared panels only change
    // cache residency, never arithmetic). Covers plain (tts) and adaln
    // (clone cond + zeros uncond) branches, with bench timings.
    use cutetts_codec::dit::euler_sample_cfg_fused;
    let wpath = match base_weights() {
        Some(p) => p,
        None => return,
    };
    let w = load_dit_weights(&wpath);
    let nth = cutetts_codec::conv::default_threads();
    // plain branches (tts)
    for i in 0..2 {
        let zc = load_vec(&format!("base_s{i:02}_zc"));
        let zu = load_vec(&format!("base_s{i:02}_zu"));
        let cond = load_vec(&format!("base_s{i:02}_cond"));
        let uncond = load_vec(&format!("base_s{i:02}_uncond"));
        let x0 = load_vec(&format!("base_s{i:02}_x0"));
        let t0 = std::time::Instant::now();
        let a = euler_sample_cfg(&w, &x0, &zc, &zu, &cond, &uncond, None, 10, -0.8, 2.0, nth);
        let ta = t0.elapsed().as_secs_f32();
        let t0 = std::time::Instant::now();
        let b = euler_sample_cfg_fused(&w, &x0, &zc, &zu, &cond, &uncond, None, 10, -0.8, 2.0, nth);
        let tb = t0.elapsed().as_secs_f32();
        assert_eq!(a.len(), b.len());
        assert!(a == b, "fused/plain mismatch s{i:02}");
        println!("fused/plain s{i:02}: bitwise equal, unfused={ta:.3}s fused={tb:.3}s (x{:.2})", ta / tb.max(1e-9));
    }
    // adaln branches (clone): cond adaln(spk), uncond adaln(zeros)
    {
        let zc = load_vec("base_ss_zc");
        let zu = load_vec("base_ss_zu");
        let cond = load_vec("base_ss_cond");
        let uncond = load_vec("base_ss_uncond");
        let spk = load_vec("base_ss_spk");
        let x0 = load_vec("base_ss_x0");
        let t0 = std::time::Instant::now();
        let a = euler_sample_cfg(&w, &x0, &zc, &zu, &cond, &uncond, Some(&spk), 10, -0.8, 2.0, nth);
        let ta = t0.elapsed().as_secs_f32();
        let t0 = std::time::Instant::now();
        let b = euler_sample_cfg_fused(&w, &x0, &zc, &zu, &cond, &uncond, Some(&spk), 10, -0.8, 2.0, nth);
        let tb = t0.elapsed().as_secs_f32();
        assert_eq!(a.len(), b.len());
        assert!(a == b, "fused/spk mismatch");
        println!("fused/spk: bitwise equal, unfused={ta:.3}s fused={tb:.3}s (x{:.2})", ta / tb.max(1e-9));
    }
    println!("FUSION bitwise gate passed");
}
