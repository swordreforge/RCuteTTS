//! AudioDiTHead weights (distill): 61 tensors / 73.6M fp32, plain Linears.
//!
//! Source: `model/CuteTTS-distill/weights/tts/model.safetensors`, `head.*`.
//! Unlike the VAE there is NO weight_norm — `nn.Linear` weights are dense
//! `[O, I]` row-major, packed directly with [`crate::gemm::pack_a`].
//! `mu_proj` is `nn.Identity` (cond_dim == hidden_dim), no tensors.
//!
//! Per-layer params: attn q/o 1024² + k/v 128×1024 (GQA 16Q/2KV, head_dim 64,
//! RoPE θ=1e4, bidirectional, seq=5) + SwiGLU MLP 3×(1024×4096) + 2×RMSNorm
//! + speaker AdaLN-Zero 6144×256 (zero-init, bias=False).

use crate::gemm::{pack_a, PackedA};
use safetensors::SafeTensors;
use std::collections::HashMap;

pub const HIDDEN: usize = 1024;
pub const FFN: usize = 4096;
pub const N_HEADS: usize = 16;
pub const N_KV: usize = 2;
pub const HEAD_DIM: usize = 64;
pub const LATENT: usize = 64;
pub const PATCH: usize = 2;
pub const SEQ: usize = 1 + PATCH + PATCH; // [mu+t+dt, cond(2), x(2)] = 5
pub const SPK: usize = 256;
pub const N_LAYERS: usize = 4;

pub struct DitLinear {
    pub w: PackedA,
    /// Padded bias (zeros for bias=False linears).
    pub b: Vec<f32>,
    pub out_dim: usize,
}

pub struct DitMlp {
    pub gate: DitLinear,
    pub up: DitLinear,
    pub down: DitLinear,
}

pub struct DitLayerW {
    pub norm1: Vec<f32>,
    pub norm2: Vec<f32>,
    pub q: DitLinear,
    pub k: DitLinear,
    pub v: DitLinear,
    pub o: DitLinear,
    pub mlp: DitMlp,
    /// speaker AdaLN-Zero: [6144, 256] -> shift/scale/gate x attn/mlp.
    pub adaln: DitLinear,
}

pub struct DitW {
    pub in_proj: DitLinear,
    pub cond_proj: DitLinear,
    pub out_proj: DitLinear,
    pub time_mlp1: DitLinear,
    pub time_mlp2: DitLinear,
    pub delta_mlp1: DitLinear,
    pub delta_mlp2: DitLinear,
    pub step_mlp1: DitLinear,
    pub step_mlp2: DitLinear,
    /// Linear(1,1024,bias=False) -> SiLU -> Linear(1024,1024,bias=False).
    pub cfg_emb0: DitLinear,
    pub cfg_emb2: DitLinear,
    pub layers: [DitLayerW; N_LAYERS],
    pub final_norm: Vec<f32>,
}

fn dense(map: &HashMap<String, (Vec<usize>, Vec<f32>)>, name: &str) -> (Vec<usize>, Vec<f32>) {
    map.get(name).unwrap_or_else(|| panic!("missing tensor {name}")).clone()
}

/// Pack a dense `[O, I]` Linear (+ optional `[O]` bias) for [`crate::gemm`].
fn lin(map: &HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str, o: usize, i: usize) -> DitLinear {
    let (s, w) = dense(map, &format!("{prefix}.weight"));
    assert_eq!(s, vec![o, i], "shape for {prefix}.weight");
    let b = match map.get(&format!("{prefix}.bias")) {
        Some((sb, bv)) => {
            assert_eq!(*sb, vec![o], "shape for {prefix}.bias");
            bv.clone()
        }
        None => vec![0.0f32; o],
    };
    let g = pack_a(&w, o, i);
    let mut bp = vec![0.0f32; g.rows];
    bp[..o].copy_from_slice(&b);
    DitLinear { w: g, b: bp, out_dim: o }
}

fn norm_w(map: &HashMap<String, (Vec<usize>, Vec<f32>)>, name: &str, dim: usize) -> Vec<f32> {
    let (s, v) = dense(map, name);
    assert_eq!(s, vec![dim], "shape for {name}");
    v
}

pub fn load_dit_weights(path: &std::path::Path) -> DitW {    let bytes = std::fs::read(path).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let mut map: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    for name in st.names() {
        if !name.starts_with("head.") {
            continue;
        }
        let tv = st.tensor(name).unwrap();
        assert_eq!(tv.dtype(), safetensors::Dtype::F32, "{name}");
        let shape = tv.shape().to_vec();
        let v: Vec<f32> = tv
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(v.len(), shape.iter().product::<usize>(), "{name}");
        map.insert(name["head.".len()..].to_string(), (shape, v));
    }
    assert_eq!(map.len(), 61, "distill head tensor count, got {}", map.len());

    let mut layers = Vec::with_capacity(N_LAYERS);
    for li in 0..N_LAYERS {
        let lb = format!("decoder.layers.{li}");
        let ab = format!("{lb}.self_attn");
        let mb = format!("{lb}.mlp");
        layers.push(DitLayerW {
            norm1: norm_w(&map, &format!("{lb}.input_layernorm.weight"), HIDDEN),
            norm2: norm_w(&map, &format!("{lb}.post_attention_layernorm.weight"), HIDDEN),
            q: lin(&map, &format!("{ab}.q_proj"), HIDDEN, HIDDEN),
            k: lin(&map, &format!("{ab}.k_proj"), N_KV * HEAD_DIM, HIDDEN),
            v: lin(&map, &format!("{ab}.v_proj"), N_KV * HEAD_DIM, HIDDEN),
            o: lin(&map, &format!("{ab}.o_proj"), HIDDEN, HIDDEN),
            mlp: DitMlp {
                gate: lin(&map, &format!("{mb}.gate_proj"), FFN, HIDDEN),
                up: lin(&map, &format!("{mb}.up_proj"), FFN, HIDDEN),
                down: lin(&map, &format!("{mb}.down_proj"), HIDDEN, FFN),
            },
            adaln: lin(&map, &format!("{lb}.speaker_adaln"), 6 * HIDDEN, SPK),
        });
    }
    DitW {
        in_proj: lin(&map, "in_proj", HIDDEN, LATENT),
        cond_proj: lin(&map, "cond_proj", HIDDEN, LATENT),
        out_proj: lin(&map, "out_proj", LATENT, HIDDEN),
        time_mlp1: lin(&map, "time_mlp.linear_1", HIDDEN, HIDDEN),
        time_mlp2: lin(&map, "time_mlp.linear_2", HIDDEN, HIDDEN),
        delta_mlp1: lin(&map, "delta_time_mlp.linear_1", HIDDEN, HIDDEN),
        delta_mlp2: lin(&map, "delta_time_mlp.linear_2", HIDDEN, HIDDEN),
        step_mlp1: lin(&map, "step_size_embedding.linear_1", HIDDEN, HIDDEN),
        step_mlp2: lin(&map, "step_size_embedding.linear_2", HIDDEN, HIDDEN),
        cfg_emb0: lin(&map, "cfg_strength_embedding.0", HIDDEN, 1),
        cfg_emb2: lin(&map, "cfg_strength_embedding.2", HIDDEN, HIDDEN),
        layers: [layers.remove(0), layers.remove(0), layers.remove(0), layers.remove(0)],
        final_norm: norm_w(&map, "decoder.norm.weight", HIDDEN),
    }
}

// ============================================================
// _predict forward (distill): x[2,64] t[] z[1024] cond[2,64]
// spk[256] dt w -> velocity[2,64].
// Mirrors AudioDiTHead._predict (diffusion_head.py:544-635).
// Activations stay [S, H] row-major; linears transpose+pad S->8
// into the 8x8 micro-kernel. Attention (5x5) is scalar.
// ============================================================

use crate::conv::default_threads;
use crate::gemm::sgemm_bias;

const ROPE_THETA: f32 = 10000.0;
const RMS_EPS: f32 = 1e-6;
const ATTN_SCALE: f32 = 1.0 / 8.0; // 1/sqrt(64)

/// Linear on row-major `[S, I]` -> `[S, O]`. Pads S to 8 cols so the
/// 8x8 micro-kernel fires; big MLP linears thread, tiny ones stay serial.
pub(crate) fn linear_rows(x: &[f32], s: usize, l: &DitLinear, nth: usize) -> Vec<f32> {
    let i = l.w.cols;
    let o = l.out_dim;
    assert_eq!(x.len(), s * i);
    assert!(s <= 8, "linear_rows handles at most 8 rows (DiT S<=5); use chunking for more");
    const T: usize = 8;
    let mut xb = vec![0.0f32; i * T];
    for ii in 0..i {
        for ss in 0..s {
            xb[ii * T + ss] = x[ss * i + ii];
        }
    }
    let ln = if (o as u64) * (i as u64) * (s as u64) >= 8_000_000 { nth } else { 1 };
    let mut c = vec![0.0f32; l.w.rows * T];
    sgemm_bias(&l.w, &xb, T, &l.b, &mut c, ln);
    let mut out = vec![0.0f32; s * o];
    for ss in 0..s {
        for oo in 0..o {
            out[ss * o + oo] = c[oo * T + ss];
        }
    }
    out
}

pub(crate) fn rms_norm_row(x: &[f32], w: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), w.len());
    let mut s = 0.0f32;
    for &v in x {
        s += v * v;
    }
    let inv = 1.0 / (s / x.len() as f32 + RMS_EPS).sqrt();
    x.iter().zip(w.iter()).map(|(&a, &b)| a * inv * b).collect()
}

fn silu_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

/// SinusoidalPosEmb(dim=1024) at scalar x (diffusion_head.py:306-320).
fn sin_emb(x: f32) -> Vec<f32> {
    const DIM: usize = HIDDEN;
    const HALF: usize = DIM / 2;
    let e = std::f32::consts::LN_10 * 4.0 / (HALF - 1) as f32; // ln(10000)/(half-1)
    let mut out = vec![0.0f32; DIM];
    for i in 0..HALF {
        let v = 1000.0 * x * (-e * i as f32).exp();
        out[i] = v.sin();
        out[HALF + i] = v.cos();
    }
    out
}

/// RoPE cos/sin tables for `seq` positions, head_dim 64, theta 1e4.
/// Layout: [pos][2*32] with emb=[f,f] (matches _apply_local_rope).
pub(crate) fn rope_tables_for(seq: usize) -> (Vec<f32>, Vec<f32>) {
    const D2: usize = HEAD_DIM / 2;
    let mut cos = vec![0.0f32; seq * HEAD_DIM];
    let mut sin = vec![0.0f32; seq * HEAD_DIM];
    for p in 0..seq {
        for i in 0..D2 {
            let f = p as f32 * ROPE_THETA.powf(-((2 * i) as f32) / HEAD_DIM as f32);
            let (s, c) = f.sin_cos();
            cos[p * HEAD_DIM + i] = c;
            cos[p * HEAD_DIM + D2 + i] = c;
            sin[p * HEAD_DIM + i] = s;
            sin[p * HEAD_DIM + D2 + i] = s;
        }
    }
    (cos, sin)
}

pub(crate) fn rope_apply(v: &[f32], pos: usize, cos: &[f32], sin: &[f32]) -> Vec<f32> {
    // v: [64]; out = v*cos + rotate_half(v)*sin
    assert_eq!(v.len(), HEAD_DIM);
    const D2: usize = HEAD_DIM / 2;
    let co = &cos[pos * HEAD_DIM..(pos + 1) * HEAD_DIM];
    let si = &sin[pos * HEAD_DIM..(pos + 1) * HEAD_DIM];
    let mut out = vec![0.0f32; HEAD_DIM];
    for i in 0..HEAD_DIM {
        let r = if i < D2 { -v[D2 + i] } else { v[i - D2] };
        out[i] = v[i] * co[i] + r * si[i];
    }
    out
}

pub(crate) fn softmax_row(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    let inv = 1.0 / s;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// One LocalTransformerLayer with precomputed AdaLN params.
/// h: [SEQ, HIDDEN] row-major, in/out. ad: 6x1024 (shift/scale/gate attn/mlp).
fn dit_layer(h: &mut [f32], lw: &DitLayerW, ad: &[f32], nth: usize, cos: &[f32], sin: &[f32]) {
    assert_eq!(ad.len(), 6 * HIDDEN);
    let (sh_a, rest) = ad.split_at(HIDDEN);
    let (sc_a, rest) = rest.split_at(HIDDEN);
    let (ga_a, rest) = rest.split_at(HIDDEN);
    let (sh_m, rest) = rest.split_at(HIDDEN);
    let (sc_m, rest) = rest.split_at(HIDDEN);
    let (ga_m, _) = rest.split_at(HIDDEN);

    // --- attention block ---
    let mut normed = vec![0.0f32; SEQ * HIDDEN];
    for s in 0..SEQ {
        let n = rms_norm_row(&h[s * HIDDEN..(s + 1) * HIDDEN], &lw.norm1);
        for d in 0..HIDDEN {
            normed[s * HIDDEN + d] = n[d] * (1.0 + sc_a[d]) + sh_a[d];
        }
    }
    let q = linear_rows(&normed, SEQ, &lw.q, nth); // [5,1024]
    let k = linear_rows(&normed, SEQ, &lw.k, nth); // [5,128]
    let v = linear_rows(&normed, SEQ, &lw.v, nth);
    // per head: rope + 5x5 scores + weighted sum (GQA: kv head = hh/8)
    let mut attn_out = vec![0.0f32; SEQ * HIDDEN];
    let mut scores = [0.0f32; SEQ * SEQ];
    let mut vv = [0.0f32; HEAD_DIM];
    for hh in 0..N_HEADS {
        let kh = hh / (N_HEADS / N_KV);
        // rope'd q/k for this head
        let mut qr = [[0.0f32; HEAD_DIM]; SEQ];
        let mut kr = [[0.0f32; HEAD_DIM]; SEQ];
        for s in 0..SEQ {
            let mut qv = [0.0f32; HEAD_DIM];
            let mut kv = [0.0f32; HEAD_DIM];
            for d in 0..HEAD_DIM {
                qv[d] = q[s * HIDDEN + hh * HEAD_DIM + d];
                kv[d] = k[s * (N_KV * HEAD_DIM) + kh * HEAD_DIM + d];
            }
            qr[s] = rope_apply(&qv, s, cos, sin).try_into().unwrap();
            kr[s] = rope_apply(&kv, s, cos, sin).try_into().unwrap();
        }
        for i in 0..SEQ {
            for j in 0..SEQ {
                let mut s = 0.0f32;
                for d in 0..HEAD_DIM {
                    s += qr[i][d] * kr[j][d];
                }
                scores[i * SEQ + j] = s * ATTN_SCALE;
            }
            softmax_row(&mut scores[i * SEQ..(i + 1) * SEQ]);
        }
        for i in 0..SEQ {
            for d in 0..HEAD_DIM {
                vv[d] = 0.0;
            }
            for j in 0..SEQ {
                let p = scores[i * SEQ + j];
                for d in 0..HEAD_DIM {
                    vv[d] += p * v[j * (N_KV * HEAD_DIM) + kh * HEAD_DIM + d];
                }
            }
            for d in 0..HEAD_DIM {
                attn_out[i * HIDDEN + hh * HEAD_DIM + d] = vv[d];
            }
        }
    }
    let attn_proj = linear_rows(&attn_out, SEQ, &lw.o, nth);
    for s in 0..SEQ {
        for d in 0..HIDDEN {
            h[s * HIDDEN + d] += (1.0 + ga_a[d]) * attn_proj[s * HIDDEN + d];
        }
    }

    // --- mlp block ---
    let mut normed2 = vec![0.0f32; SEQ * HIDDEN];
    for s in 0..SEQ {
        let n = rms_norm_row(&h[s * HIDDEN..(s + 1) * HIDDEN], &lw.norm2);
        for d in 0..HIDDEN {
            normed2[s * HIDDEN + d] = n[d] * (1.0 + sc_m[d]) + sh_m[d];
        }
    }
    let gate = linear_rows(&normed2, SEQ, &lw.mlp.gate, nth);
    let up = linear_rows(&normed2, SEQ, &lw.mlp.up, nth);
    let mut act = vec![0.0f32; SEQ * FFN];
    for i in 0..act.len() {
        let g = gate[i] / (1.0 + (-gate[i]).exp());
        act[i] = g * up[i];
    }
    let down = linear_rows(&act, SEQ, &lw.mlp.down, nth);
    for s in 0..SEQ {
        for d in 0..HIDDEN {
            h[s * HIDDEN + d] += (1.0 + ga_m[d]) * down[s * HIDDEN + d];
        }
    }
}

/// TimestepEmbedding: Linear -> SiLU -> Linear (bias=True).
fn time_mlp(x: &[f32], l1: &DitLinear, l2: &DitLinear, nth: usize) -> Vec<f32> {
    let h = linear_rows(x, 1, l1, nth);
    let a = silu_vec(&h);
    linear_rows(&a, 1, l2, nth)
}

/// cfg_strength path: Linear(1->H, no bias) -> SiLU -> Linear(H->H, no bias).
fn cfg_emb(w_norm: f32, e0: &DitLinear, e2: &DitLinear, nth: usize) -> Vec<f32> {
    let h = linear_rows(&[w_norm], 1, e0, nth);
    let a = silu_vec(&h);
    linear_rows(&a, 1, e2, nth)
}

/// Full _predict: x[2,64], t, z[1024], cond[2,64], dt, spk[256], w -> [2,64].
pub fn predict(
    w: &DitW,
    x: &[f32],
    t: f32,
    z: &[f32],
    cond: &[f32],
    dt: f32,
    spk: &[f32],
    cfg_w: f32,
    nth: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), PATCH * LATENT);
    assert_eq!(z.len(), HIDDEN);
    assert_eq!(cond.len(), PATCH * LATENT);
    assert_eq!(spk.len(), SPK);

    let xh = linear_rows(x, PATCH, &w.in_proj, nth); // [2,1024]
    let ch = linear_rows(cond, PATCH, &w.cond_proj, nth);
    let t_emb = time_mlp(&sin_emb(t), &w.time_mlp1, &w.time_mlp2, nth);
    let dt_emb = time_mlp(&sin_emb(0.0), &w.delta_mlp1, &w.delta_mlp2, nth);
    let step_emb = time_mlp(&sin_emb(dt), &w.step_mlp1, &w.step_mlp2, nth);
    let strength = cfg_emb(cfg_w / 5.0, &w.cfg_emb0, &w.cfg_emb2, nth);
    // mu = z + t + dt + step + cfg
    let mut mu = vec![0.0f32; HIDDEN];
    for d in 0..HIDDEN {
        mu[d] = z[d] + t_emb[d] + dt_emb[d] + step_emb[d] + strength[d];
    }
    // seq = [mu, cond(2), x(2)]
    let mut seq = vec![0.0f32; SEQ * HIDDEN];
    seq[..HIDDEN].copy_from_slice(&mu);
    seq[HIDDEN..3 * HIDDEN].copy_from_slice(&ch);
    seq[3 * HIDDEN..].copy_from_slice(&xh);

    // speaker adaln once per predict: [6144]
    let (cos, sin) = rope_tables_for(SEQ);
    for lw in &w.layers {
        let adl = linear_rows(spk, 1, &lw.adaln, nth);
        dit_layer(&mut seq, lw, &adl, nth, &cos, &sin);
    }
    for s in 0..SEQ {
        let n = rms_norm_row(&seq[s * HIDDEN..(s + 1) * HIDDEN], &w.final_norm);
        seq[s * HIDDEN..(s + 1) * HIDDEN].copy_from_slice(&n);
    }
    // velocity = out_proj(last 2 tokens)
    linear_rows(&seq[3 * HIDDEN..], PATCH, &w.out_proj, nth)
}

/// [`predict`] with default thread count.
pub fn predict_default(
    w: &DitW,
    x: &[f32],
    t: f32,
    z: &[f32],
    cond: &[f32],
    dt: f32,
    spk: &[f32],
    cfg_w: f32,
) -> Vec<f32> {
    predict(w, x, t, z, cond, dt, spk, cfg_w, default_threads())
}

/// Precomputed per-sample conditions (torch recomputes these every Euler
/// step; t varies per step so only t_emb is per-step).
/// adaln is per-layer (each layer has its own speaker_adaln matrix).
pub struct SampleCond {
    pub dt_emb: Vec<f32>,
    pub step_emb: Vec<f32>,
    pub cfg_emb: Vec<f32>,
    pub adalns: Vec<Vec<f32>>, // [layer][6144]
}

/// Prologue of sample(): conditions that are constant across Euler steps.
/// t_emb is NOT included (depends on step t).
pub fn sample_prologue(w: &DitW, spk: &[f32], dt: f32, cfg_w: f32, nth: usize) -> SampleCond {
    let dt_emb = time_mlp(&sin_emb(0.0), &w.delta_mlp1, &w.delta_mlp2, nth);
    let step_emb = time_mlp(&sin_emb(dt), &w.step_mlp1, &w.step_mlp2, nth);
    let cfg_emb = cfg_emb(cfg_w / 5.0, &w.cfg_emb0, &w.cfg_emb2, nth);
    let adalns = w.layers.iter().map(|lw| linear_rows(spk, 1, &lw.adaln, nth)).collect();
    SampleCond { dt_emb, step_emb, cfg_emb, adalns }
}

/// One _predict with precomputed conditions (mirrors _predict with cache hit).
fn predict_cached(
    w: &DitW,
    x: &[f32],
    t: f32,
    z: &[f32],
    cond: &[f32],
    sc: &SampleCond,
    nth: usize,
    cos: &[f32],
    sin: &[f32],
) -> Vec<f32> {
    let xh = linear_rows(x, PATCH, &w.in_proj, nth);
    let ch = linear_rows(cond, PATCH, &w.cond_proj, nth);
    let t_emb = time_mlp(&sin_emb(t), &w.time_mlp1, &w.time_mlp2, nth);
    let mut mu = vec![0.0f32; HIDDEN];
    for d in 0..HIDDEN {
        mu[d] = z[d] + t_emb[d] + sc.dt_emb[d] + sc.step_emb[d] + sc.cfg_emb[d];
    }
    let mut seq = vec![0.0f32; SEQ * HIDDEN];
    seq[..HIDDEN].copy_from_slice(&mu);
    seq[HIDDEN..3 * HIDDEN].copy_from_slice(&ch);
    seq[3 * HIDDEN..].copy_from_slice(&xh);
    for (lw, adl) in w.layers.iter().zip(sc.adalns.iter()) {
        dit_layer(&mut seq, lw, adl, nth, cos, sin);
    }
    for s in 0..SEQ {
        let n = rms_norm_row(&seq[s * HIDDEN..(s + 1) * HIDDEN], &w.final_norm);
        seq[s * HIDDEN..(s + 1) * HIDDEN].copy_from_slice(&n);
    }
    linear_rows(&seq[3 * HIDDEN..], PATCH, &w.out_proj, nth)
}

/// Distill Euler sample: 4 uniform steps, single branch (cfg=0).
/// Mirrors _euler_sample_core with step_size embedding (dt=1/steps).
/// x0: initial noise [2,64] (caller seeds RNG — torch.randn in sample()).
pub fn euler_sample(
    w: &DitW,
    x0: &[f32],
    z: &[f32],
    cond: &[f32],
    spk: &[f32],
    steps: usize,
    cfg_w: f32,
    nth: usize,
) -> Vec<f32> {
    assert_eq!(x0.len(), PATCH * LATENT);
    let dt = 1.0 / steps as f32;
    let sc = sample_prologue(w, spk, dt, cfg_w, nth);
    let (cos, sin) = rope_tables_for(SEQ);
    let mut x = x0.to_vec();
    for step in 0..steps {
        let t = step as f32 * dt;
        let v = predict_cached(w, &x, t, z, cond, &sc, nth, &cos, &sin);
        for i in 0..x.len() {
            x[i] += v[i] * dt;
        }
    }
    x
}
