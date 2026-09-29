//! AudioVAE CausalEncoder (reference path for voice_clone in pure Rust).
//!
//! Source: `model/CuteTTS/weights/audio_vae/model.safetensors`, `encoder.*`
//! (118 tensors; `fc_logvar` skipped — sigma posterior, inference uses
//! posterior mode = mu only). All fp32 + weight_norm (fused at load).
//!
//! Forward (24k mono wave -> mu [frames, 64]):
//! preprocess right-pad to 1920 multiple -> conv k7 (1->128) ->
//! 4x [3x ResUnit(Snake -> dw k7 d=1/3/9 -> Snake -> pw k1 -> +res)
//!      + Snake + DENSE strided conv (k=2s, s=3/5/8/16)] ->
//! fc_mu k3 -> [64, F] -> rows [F, 64].
//! Strided convs go im2col + sgemm (like the speaker encoder); the rest
//! reuses the proven causal kernels.

use crate::conv::{causal_conv1d_par, default_threads, fuse_weight_norm};
use crate::gemm::{pack_a, sgemm_bias, PackedA};
use crate::snake::snake1d_nth as snake_par;
use safetensors::SafeTensors;
use std::collections::HashMap;

pub const ENC_RATES: [usize; 4] = [3, 5, 8, 16];
pub const ENC_DIMS: [usize; 4] = [128, 256, 512, 1024]; // block input dims
pub const ENC_OUTS: [usize; 4] = [256, 512, 1024, 2048];
pub const HOP: usize = 1920;

pub struct EncResUnit {
    pub a0: Vec<f32>,
    pub dw_w: Vec<f32>,
    pub dw_b: Vec<f32>,
    pub d: usize,
    pub a1: Vec<f32>,
    pub pw_w: Vec<f32>,
    pub pw_b: Vec<f32>,
    pub dim: usize,
}

pub struct EncStage {
    pub res: [EncResUnit; 3],
    pub snake_a: Vec<f32>,
    pub strided: PackedA, // [out, in*2s]
    pub strided_b: Vec<f32>,
    pub in_c: usize,
    pub out_c: usize,
    pub stride: usize,
}

pub struct VaeEncW {
    pub front_w: Vec<f32>,
    pub front_b: Vec<f32>,
    pub stages: [EncStage; 4],
    pub fc_w: Vec<f32>,
    pub fc_b: Vec<f32>,
}

fn take(map: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, name: &str) -> (Vec<usize>, Vec<f32>) {
    map.remove(name).unwrap_or_else(|| panic!("missing tensor {name}"))
}

fn fused(map: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str) -> (Vec<usize>, Vec<f32>) {
    let (sg, g) = take(map, &format!("{prefix}.weight_g"));
    let (sv, v) = take(map, &format!("{prefix}.weight_v"));
    assert_eq!(sg, vec![sv[0], 1, 1], "g shape for {prefix}");
    let (o, i, k) = (sv[0], sv[1], sv[2]);
    (vec![o, i, k], fuse_weight_norm(&g, &v, o, i, k))
}

fn bias_of(map: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str) -> Vec<f32> {
    take(map, &format!("{prefix}.bias")).1
}

fn alpha_of(map: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str) -> Vec<f32> {
    let (s, v) = take(map, &format!("{prefix}.alpha"));
    assert_eq!(s.len(), 3);
    v
}

pub fn load_vae_enc_weights(path: &std::path::Path) -> VaeEncW {
    let bytes = std::fs::read(path).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let mut map: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    for name in st.names() {
        if !name.starts_with("encoder.") || name.starts_with("encoder.fc_logvar") {
            continue;
        }
        let tv = st.tensor(name).unwrap();
        assert_eq!(tv.dtype(), safetensors::Dtype::F32, "{name}");
        let shape = tv.shape().to_vec();
        let v: Vec<f32> = tv.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(v.len(), shape.iter().product::<usize>(), "{name}");
        map.insert(name.to_string(), (shape, v));
    }
    let (s0, front_w) = fused(&mut map, "encoder.block.0");
    assert_eq!(s0, vec![128, 1, 7]);
    let front_b = bias_of(&mut map, "encoder.block.0");

    let mut stages = Vec::with_capacity(4);
    for (si, (dim, outc, stride)) in ENC_DIMS.iter().zip(ENC_OUTS.iter()).zip(ENC_RATES.iter()).map(|((a, b), c)| (*a, *b, *c)).enumerate() {
        let base = format!("encoder.block.{}", si + 1);
        let mut res = Vec::with_capacity(3);
        for (ri, &d) in [1usize, 3, 9].iter().enumerate() {
            let rb = format!("{base}.block.{ri}");
            let (sd, dw_w) = fused(&mut map, &format!("{rb}.block.1"));
            assert_eq!(sd, vec![dim, 1, 7], "dw {rb}");
            let (sp, pw_w) = fused(&mut map, &format!("{rb}.block.3"));
            assert_eq!(sp, vec![dim, dim, 1], "pw {rb}");
            res.push(EncResUnit {
                a0: alpha_of(&mut map, &format!("{rb}.block.0")),
                dw_w,
                dw_b: bias_of(&mut map, &format!("{rb}.block.1")),
                d,
                a1: alpha_of(&mut map, &format!("{rb}.block.2")),
                pw_w,
                pw_b: bias_of(&mut map, &format!("{rb}.block.3")),
                dim,
            });
        }
        let snake_a = alpha_of(&mut map, &format!("{base}.block.3"));
        assert_eq!(snake_a.len(), dim);
        let (ss, st_w) = fused(&mut map, &format!("{base}.block.4"));
        assert_eq!(ss, vec![outc, dim, 2 * stride], "strided {base}");
        // flatten [O,I,K] -> [O, I*K] row-major for im2col GEMM
        let strided = pack_a(&st_w, outc, dim * 2 * stride);
        let strided_b = bias_of(&mut map, &format!("{base}.block.4"));
        stages.push(EncStage { res: [res.remove(0), res.remove(0), res.remove(0)], snake_a, strided, strided_b, in_c: dim, out_c: outc, stride });
    }
    let (sf, fc_w) = fused(&mut map, "encoder.fc_mu");
    assert_eq!(sf, vec![64, 2048, 3]);
    let fc_b = bias_of(&mut map, "encoder.fc_mu");
    assert!(map.is_empty(), "leftover: {:?}", map.keys().collect::<Vec<_>>());
    VaeEncW {
        front_w,
        front_b,
        stages: [stages.remove(0), stages.remove(0), stages.remove(0), stages.remove(0)],
        fc_w,
        fc_b,
    }
}

/// Dense strided causal conv via im2col + sgemm.
/// torch: left-pad 2p zeros, plain stride-s conv with FLOOR output length
/// (floor((T + 2p - k)/s) + 1, like nn.Conv1d). Hop-aligned inputs still land
/// exact (T=3N -> N, T=5N -> N); the floor only absorbs the pad remainder.
fn strided_conv(x: &[f32], t: usize, st: &EncStage, nth: usize) -> Vec<f32> {
    let (s, o, i, k) = (st.stride, st.out_c, st.in_c, 2 * st.stride);
    let p = (s + 1) / 2; // ceil(s/2)
    let out_t = (t + 2 * p - k) / s + 1;
    let ik = i * k;
    let mut cols = vec![0.0f32; ik * out_t];
    for ti in 0..out_t {
        for c in 0..i {
            for kk in 0..k {
                let src = ti as isize * s as isize + kk as isize - 2 * p as isize;
                let v = if src < 0 { 0.0 } else { x[c * t + src as usize] };
                cols[(c * k + kk) * out_t + ti] = v;
            }
        }
    }
    let mut c = vec![0.0f32; st.strided.rows * out_t];
    let ln = if (o as u64) * (ik as u64) * (out_t as u64) >= 8_000_000 { nth } else { 1 };
    sgemm_bias(&st.strided, &cols, out_t, &st.strided_b, &mut c, ln);
    c.truncate(o * out_t);
    c
}

/// Encode 24k mono wave -> mu [frames, 64] (posterior mode).
pub fn vae_encode(w: &VaeEncW, wave: &[f32], nth: usize) -> Vec<f32> {
    // preprocess: right-pad to hop multiple (mirrors AudioVAE.preprocess)
    let mut x = wave.to_vec();
    let rem = x.len() % HOP;
    if rem != 0 {
        x.resize(x.len() + HOP - rem, 0.0);
    }
    let mut h = causal_conv1d_par(&x, 1, x.len(), &w.front_w, &w.front_b, 128, 7, 1, 1, 3, nth);
    let mut t = x.len();
    for st in &w.stages {
        for ru in &st.res {
            let mut r = h.clone();
            snake_par(&mut r, ru.dim, t, &ru.a0, nth);
            r = causal_conv1d_par(&r, ru.dim, t, &ru.dw_w, &ru.dw_b, ru.dim, 7, ru.d, ru.dim, 3 * ru.d, nth);
            snake_par(&mut r, ru.dim, t, &ru.a1, nth);
            r = causal_conv1d_par(&r, ru.dim, t, &ru.pw_w, &ru.pw_b, ru.dim, 1, 1, 1, 0, nth);
            h = h.iter().zip(r.iter()).map(|(a, b)| a + b).collect();
        }
        snake_par(&mut h, st.in_c, t, &st.snake_a, nth);
        h = strided_conv(&h, t, st, nth);
        t /= st.stride;
    }
    // fc_mu k3 -> [64, T]
    let mu = causal_conv1d_par(&h, 2048, t, &w.fc_w, &w.fc_b, 64, 3, 1, 1, 1, nth);
    // rows [T, 64]
    let mut out = vec![0.0f32; t * 64];
    for f in 0..t {
        for d in 0..64 {
            out[f * 64 + d] = mu[d * t + f];
        }
    }
    out
}

/// [`vae_encode`] with default thread count.
pub fn vae_encode_default(w: &VaeEncW, wave: &[f32]) -> Vec<f32> {
    vae_encode(w, wave, default_threads())
}
