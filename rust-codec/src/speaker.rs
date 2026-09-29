//! FbankECAPAStudent speaker encoder (voice_clone reference path).
//!
//! Source: `model/CuteTTS-distill/weights/speaker_encoder/model.safetensors`
//! (221 tensors, all fp32, ~25M params). Production runs fp32
//! (`speaker_encoder.float()`), so gates are tight.
//!
//! Pipeline: 16k wave -> reflect-pad STFT (512/400/160, Hann periodic,
//! center) -> power -> mel[80,257] -> log(clamp 1e-6) -> InstanceNorm ->
//! conv k5 (80->1152) -> 3x SERes2Block (1152, k3 d=2/3/4, scale 8,
//! conv->relu->BN order!) -> cat3 (3456) -> conv k1 -> relu ->
//! AttentiveStatsPool (global context) -> BN (eval) -> Linear (6912->256)
//! -> L2 normalize. All BNs use running stats (eval); eps default 1e-5.

use crate::gemm::{pack_a, sgemm_bias, PackedA};
use safetensors::SafeTensors;
use std::collections::HashMap;

// ---------- frontend ----------

const SR: usize = 16000;
const N_FFT: usize = 512;
const WIN: usize = 400;
const HOP: usize = 160;
const N_MELS: usize = 80;
const N_FREQ: usize = N_FFT / 2 + 1; // 257
const MEL_EPS: f32 = 1e-6;

/// Periodic Hann (torch.hann_window default periodic=True).
fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()) as f32)
        .collect()
}

fn hz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

fn mel_to_hz(m: f64) -> f64 {
    700.0 * (10.0f64.powf(m / 2595.0) - 1.0)
}

/// Exact replica of `_mel_filter_bank` (f64 intermediates, cast at the end).
fn mel_bank() -> Vec<f32> {
    let all: Vec<f64> = (0..N_FREQ).map(|i| i as f64 * (SR / 2) as f64 / (N_FREQ - 1) as f64).collect();
    let mmin = hz_to_mel(0.0);
    let mmax = hz_to_mel((SR / 2) as f64);
    let mpts: Vec<f64> = (0..N_MELS + 2).map(|i| mmin + (mmax - mmin) * i as f64 / (N_MELS + 1) as f64).collect();
    let fpts: Vec<f64> = mpts.iter().map(|&m| mel_to_hz(m)).collect();
    let mut bank = vec![0.0f32; N_MELS * N_FREQ];
    for m in 0..N_MELS {
        let (left, center, right) = (fpts[m], fpts[m + 1], fpts[m + 2]);
        for (f, af) in all.iter().enumerate() {
            let up = (af - left) / (center - left).max(1e-8);
            let down = (right - af) / (right - center).max(1e-8);
            bank[m * N_FREQ + f] = up.min(down).max(0.0) as f32;
        }
    }
    bank
}

/// Iterative radix-2 forward FFT, unnormalized (like torch, normalized=False).
/// re/im modified in place, len must be a power of two.
fn fft(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    assert_eq!(im.len(), n);
    assert!(n.is_power_of_two());
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let (s, c) = ang.sin_cos();
        let (wr, wi) = (c as f32, s as f32);
        let half = len / 2;
        let mut i = 0;
        while i < n {
            let (mut vr, mut vi) = (1.0f32, 0.0f32);
            for k in 0..half {
                let a = re[i + k];
                let b = im[i + k];
                let cc = re[i + k + half];
                let dd = im[i + k + half];
                // t = w * (c+id)
                let tr = vr * cc - vi * dd;
                let ti = vr * dd + vi * cc;
                re[i + k] = a + tr;
                im[i + k] = b + ti;
                re[i + k + half] = a - tr;
                im[i + k + half] = b - ti;
                let nvr = vr * wr - vi * wi;
                vi = vr * wi + vi * wr;
                vr = nvr;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Log-mel frontend: wave[T] @16k -> [80, frames]. Reflect padding mirrors
/// torch.stft(center=True); the differential test decides if this matches.
pub fn log_mel(wave: &[f32], window: &[f32], bank: &[f32]) -> Vec<f32> {
    let t = wave.len();
    let pad = N_FFT / 2;
    // reflect-pad: left[i] = x[pad-i], right = x[T-2-j]
    let mut pw = vec![0.0f32; t + 2 * pad];
    for i in 0..pad {
        pw[i] = wave[pad - i];
        pw[t + pad + i] = wave[t - 2 - i];
    }
    pw[pad..pad + t].copy_from_slice(wave);
    let nf = 1 + t / HOP;
    let mut power = vec![0.0f32; N_FREQ * nf];
    let mut re = vec![0.0f32; N_FFT];
    let mut im = vec![0.0f32; N_FFT];
    // torch.stft centers a short window inside the n_fft frame:
    // FFT bins [woff, woff+WIN) hold signal [base+woff, base+woff+WIN).
    let woff = (N_FFT - WIN) / 2;
    for f in 0..nf {
        let base = f * HOP;
        for n in 0..N_FFT {
            re[n] = 0.0;
            im[n] = 0.0;
        }
        for n in 0..WIN {
            re[woff + n] = pw[base + woff + n] * window[n];
        }
        fft(&mut re, &mut im);
        for k in 0..N_FREQ {
            power[k * nf + f] = re[k] * re[k] + im[k] * im[k];
        }
    }
    // mel = bank[80,257] @ power[257,nf] -> [80,nf]; log(clamp)
    let mut mel = vec![0.0f32; N_MELS * nf];
    for m in 0..N_MELS {
        for f in 0..nf {
            let mut acc = 0.0f32;
            for k in 0..N_FREQ {
                acc += bank[m * N_FREQ + k] * power[k * nf + f];
            }
            mel[m * nf + f] = acc.max(MEL_EPS).ln();
        }
    }
    mel
}

// ---------- inference ops (all eval, fp32) ----------

pub struct Bn {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub mean: Vec<f32>,
    pub var: Vec<f32>,
}

pub struct SpkConv {
    pub w: PackedA, // dense [O, I*K] im2col GEMM
    pub b: Vec<f32>,
    pub out_c: usize,
    pub k: usize,
    pub pad: usize,
    pub dilation: usize,
}

pub struct SpkRes2 {
    pub convs: Vec<SpkConv>, // 7x [144,144,3,d]
    pub bns: Vec<Bn>,       // 7x [144]
}

pub struct SpkSeBlock {
    pub c1: SpkConv, // k1 in->out
    pub b1: Bn,
    pub res2: SpkRes2,
    pub c2: SpkConv, // k1
    pub b2: Bn,
    pub se1w: PackedA, // [128, C] GEMM (with bias)
    pub se1b: Vec<f32>,
    pub se2w: PackedA, // [C, 128]
    pub se2b: Vec<f32>,
    pub channels: usize,
}

pub struct SpeakerW {
    pub window: Vec<f32>,
    pub mel_bank: Vec<f32>,
    pub l1: SpkConv,
    pub l1bn: Bn,
    pub blocks: [SpkSeBlock; 3],
    pub cat_conv: SpkConv, // k1 3456->3456
    pub pool_w1: PackedA,  // [128, 10368]
    pub pool_b1: Vec<f32>,
    pub pool_w2: PackedA, // [3456, 128]
    pub pool_b2: Vec<f32>,
    pub final_bn: Bn, // [6912]
    pub lin_w: PackedA, // [256, 6912]
    pub lin_b: Vec<f32>,
}

fn bn_eval(x: &[f32], c: usize, t: usize, bn: &Bn) -> Vec<f32> {
    assert_eq!(x.len(), c * t);
    let mut out = vec![0.0f32; c * t];
    for ch in 0..c {
        let inv = 1.0 / (bn.var[ch] + 1e-5).sqrt();
        for ti in 0..t {
            let i = ch * t + ti;
            out[i] = (x[i] - bn.mean[ch]) * inv * bn.w[ch] + bn.b[ch];
        }
    }
    out
}

fn relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Plain (non-causal) conv1d stride 1 via im2col + sgemm.
/// Symmetric zero padding, dilation aware.
fn plain_conv(x: &[f32], c_in: usize, t: usize, cv: &SpkConv, nth: usize) -> Vec<f32> {
    assert_eq!(x.len(), c_in * t);
    let o = cv.out_c;
    let k = cv.k;
    let ik = c_in * k;
    // im2col: cols[t] of c_in*k taps (symmetric pad, dilation).
    // All kernels are odd with symmetric torch padding, so the left offset
    // ((k-1)/2 * d) is exact; verified by the differential test.
    debug_assert_eq!(cv.pad, ((k - 1) / 2) * cv.dilation);
    let mut cols = vec![0.0f32; ik * t];
    for ti in 0..t {
        for c in 0..c_in {
            for kk in 0..k {
                let src = ti as isize + (kk as isize - (k as isize - 1) / 2) * cv.dilation as isize;
                // NOTE: (k-1)/2 floors; all our kernels are odd with symmetric
                // padding, and torch padding is symmetric, so left offset is
                // exact. Veriy via differential test.
                let v = if src < 0 || src >= t as isize { 0.0 } else { x[c * t + src as usize] };
                cols[(c * k + kk) * t + ti] = v;
            }
        }
    }
    // GEMM [O, ik] x [ik, t]
    let mut c = vec![0.0f32; cv.w.rows * t];
    // bias padded inside SpkConv build (zeros beyond O)
    sgemm_bias(&cv.w, &cols, t, &cv.b, &mut c, nth);
    let mut out = vec![0.0f32; o * t];
    for oo in 0..o {
        out[oo * t..(oo + 1) * t].copy_from_slice(&c[oo * t..oo * t + t]);
    }
    out
}

/// conv -> relu -> bn (note order! matches Res2Conv1dReluBn/Conv1dReluBn).
fn conv_relu_bn(x: &[f32], c_in: usize, t: usize, cv: &SpkConv, bn: &Bn, nth: usize) -> Vec<f32> {
    let mut y = plain_conv(x, c_in, t, cv, nth);
    relu_inplace(&mut y);
    bn_eval(&y, cv.out_c, t, bn)
}

fn res2_block(x: &[f32], c: usize, t: usize, r: &SpkRes2, nth: usize) -> Vec<f32> {
    const SCALE: usize = 8;
    let w = c / SCALE;
    assert_eq!(c, w * SCALE);
    let mut outs: Vec<Vec<f32>> = Vec::with_capacity(SCALE);
    let mut chunk = x[..w * t].to_vec();
    for i in 0..7 {
        if i > 0 {
            let s = &x[i * w * t..(i + 1) * w * t];
            for (a, b) in chunk.iter_mut().zip(s.iter()) {
                *a += *b;
            }
        }
        let y = plain_conv(&chunk, w, t, &r.convs[i], nth);
        let mut y2 = y;
        relu_inplace(&mut y2);
        chunk = bn_eval(&y2, w, t, &r.bns[i]);
        outs.push(chunk.clone());
    }
    outs.push(x[7 * w * t..8 * w * t].to_vec());
    let mut out = vec![0.0f32; c * t];
    for (i, o) in outs.iter().enumerate() {
        out[i * w * t..(i + 1) * w * t].copy_from_slice(o);
    }
    out
}

fn se_block(x: &[f32], c: usize, t: usize, b: &SpkSeBlock, nth: usize) -> Vec<f32> {
    let h1 = conv_relu_bn(x, c, t, &b.c1, &b.b1, nth);
    let h2 = res2_block(&h1, c, t, &b.res2, nth);
    let h3 = conv_relu_bn(&h2, c, t, &b.c2, &b.b2, nth);
    // SE: mean over time -> fc -> relu -> fc -> sigmoid -> scale
    let mut mean = vec![0.0f32; c];
    for ch in 0..c {
        let mut s = 0.0f32;
        for ti in 0..t {
            s += h3[ch * t + ti];
        }
        mean[ch] = s / t as f32;
    }
    let mut s1 = vec![0.0f32; b.se1w.rows];
    sgemm_bias(&b.se1w, &mean, 1, &b.se1b, &mut s1, 1);
    s1.truncate(128);
    relu_inplace(&mut s1);
    let mut s2 = vec![0.0f32; b.se2w.rows];
    sgemm_bias(&b.se2w, &s1, 1, &b.se2b, &mut s2, 1);
    s2.truncate(c);
    let mut out = vec![0.0f32; c * t];
    for ch in 0..c {
        let g = 1.0 / (1.0 + (-s2[ch]).exp());
        for ti in 0..t {
            out[ch * t + ti] = h3[ch * t + ti] * g;
        }
    }
    // shortcut: None (in == out == channels for layers 2/3/4)
    for (o, r) in out.iter_mut().zip(x.iter()) {
        *o += *r;
    }
    out
}

/// Full forward: wave[T] @16k -> embedding[256] (L2-normalized).
pub fn speaker_forward(w: &SpeakerW, wave: &[f32], nth: usize) -> Vec<f32> {
    let nf = 1 + wave.len() / HOP;
    let mel = log_mel(wave, &w.window, &w.mel_bank);
    // instance norm (no affine)
    let mut x = vec![0.0f32; N_MELS * nf];
    for m in 0..N_MELS {
        let mut s = 0.0f32;
        for f in 0..nf {
            s += mel[m * nf + f];
        }
        let mean = s / nf as f32;
        let mut v = 0.0f32;
        for f in 0..nf {
            let d = mel[m * nf + f] - mean;
            v += d * d;
        }
        let inv = 1.0 / (v / nf as f32 + 1e-6).sqrt();
        for f in 0..nf {
            x[m * nf + f] = (mel[m * nf + f] - mean) * inv;
        }
    }
    let h1 = conv_relu_bn(&x, N_MELS, nf, &w.l1, &w.l1bn, nth);
    let mut outs = Vec::with_capacity(3);
    let mut h = h1;
    for b in &w.blocks {
        h = se_block(&h, 1152, nf, b, nth);
        outs.push(h.clone());
    }
    let c = 1152;
    let mut cat = vec![0.0f32; 3 * c * nf];
    for (i, o) in outs.iter().enumerate() {
        cat[i * c * nf..(i + 1) * c * nf].copy_from_slice(o);
    }
    embed_tail(w, &cat, nf, nth)
}

/// Post-cat tail (shared by forward; exposed for tests).
pub fn embed_tail(w: &SpeakerW, cat: &[f32], nf: usize, nth: usize) -> Vec<f32> {
    let c = 1152;
    let mut h = plain_conv(&cat, 3 * c, nf, &w.cat_conv, nth);
    relu_inplace(&mut h);
    // attentive stats pool (global context)
    let cc = 3 * c; // 3456
    let mut mean = vec![0.0f32; cc];
    let mut stdv = vec![0.0f32; cc];
    for ch in 0..cc {
        let mut s = 0.0f32;
        for ti in 0..nf {
            s += h[ch * nf + ti];
        }
        let m = s / nf as f32;
        mean[ch] = m;
        let mut v = 0.0f32;
        for ti in 0..nf {
            let d = h[ch * nf + ti] - m;
            v += d * d;
        }
        // torch.var default unbiased=True (Bessel)
        stdv[ch] = (v / (nf - 1) as f32 + 1e-10).sqrt();
    }
    let ci = 3 * cc; // 10368
    let mut xin = vec![0.0f32; ci * nf];
    for ti in 0..nf {
        for ch in 0..cc {
            xin[ch * nf + ti] = h[ch * nf + ti];
            xin[(cc + ch) * nf + ti] = mean[ch];
            xin[(2 * cc + ch) * nf + ti] = stdv[ch];
        }
    }
    let mut a1 = vec![0.0f32; w.pool_w1.rows * nf];
    sgemm_bias(&w.pool_w1, &xin, nf, &w.pool_b1, &mut a1, nth);
    a1.truncate(128 * nf);
    for v in a1.iter_mut() {
        *v = v.tanh();
    }
    let mut a2 = vec![0.0f32; w.pool_w2.rows * nf];
    sgemm_bias(&w.pool_w2, &a1, nf, &w.pool_b2, &mut a2, nth);
    a2.truncate(cc * nf);
    // softmax over time per channel
    let mut alpha = vec![0.0f32; cc * nf];
    for ch in 0..cc {
        let mut m = f32::NEG_INFINITY;
        for ti in 0..nf {
            m = m.max(a2[ch * nf + ti]);
        }
        let mut s = 0.0f32;
        for ti in 0..nf {
            let e = (a2[ch * nf + ti] - m).exp();
            alpha[ch * nf + ti] = e;
            s += e;
        }
        for ti in 0..nf {
            alpha[ch * nf + ti] /= s;
        }
    }
    let mut pooled = vec![0.0f32; 2 * cc];
    for ch in 0..cc {
        let mut m = 0.0f32;
        let mut q = 0.0f32;
        for ti in 0..nf {
            m += alpha[ch * nf + ti] * h[ch * nf + ti];
            q += alpha[ch * nf + ti] * h[ch * nf + ti] * h[ch * nf + ti];
        }
        pooled[ch] = m;
        pooled[cc + ch] = (q - m * m).max(1e-9).sqrt();
    }
    let normed = bn_eval(&pooled, 2 * cc, 1, &w.final_bn);
    let mut emb = vec![0.0f32; w.lin_w.rows];
    sgemm_bias(&w.lin_w, &normed, 1, &w.lin_b, &mut emb, 1);
    emb.truncate(256);
    let n = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
    emb.iter_mut().for_each(|v| *v /= n);
    emb
}

// ---------- weights ----------

fn take(map: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, name: &str) -> (Vec<usize>, Vec<f32>) {
    map.remove(name).unwrap_or_else(|| panic!("missing tensor {name}"))
}

fn pack_conv(map: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str, o: usize, i: usize, k: usize) -> (PackedA, Vec<f32>) {
    let (s, w) = take(map, &format!("{prefix}.weight"));
    assert_eq!(s, vec![o, i, k], "shape for {prefix}.weight");
    // flatten [O,I,K] -> [O, I*K] row-major for im2col GEMM
    let g = pack_a(&w, o, i * k);
    let (sb, bv) = take(map, &format!("{prefix}.bias"));
    assert_eq!(sb, vec![o]);
    let mut bp = vec![0.0f32; g.rows];
    bp[..o].copy_from_slice(&bv);
    (g, bp)
}

fn take_bn(map: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str, c: usize) -> Bn {
    let (_, w) = take(map, &format!("{prefix}.weight"));
    let (_, b) = take(map, &format!("{prefix}.bias"));
    let (_, mean) = take(map, &format!("{prefix}.running_mean"));
    let (_, var) = take(map, &format!("{prefix}.running_var"));
    assert_eq!(w.len(), c);
    Bn { w, b, mean, var }
}

fn pack_linear(map: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str, o: usize, i: usize) -> (PackedA, Vec<f32>) {
    let (s, w) = take(map, &format!("{prefix}.weight"));
    assert_eq!(s, vec![o, i]);
    let g = pack_a(&w, o, i);
    let (sb, bv) = take(map, &format!("{prefix}.bias"));
    assert_eq!(sb, vec![o]);
    let mut bp = vec![0.0f32; g.rows];
    bp[..o].copy_from_slice(&bv);
    (g, bp)
}

fn load_spkconv(map: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str, o: usize, i: usize, k: usize, pad: usize, dilation: usize) -> SpkConv {
    let (w, b) = pack_conv(map, prefix, o, i, k);
    SpkConv { w, b, out_c: o, k, pad, dilation }
}

pub fn load_speaker_weights(path: &std::path::Path) -> SpeakerW {    let bytes = std::fs::read(path).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let mut map: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    for name in st.names() {
        if name.ends_with("num_batches_tracked") {
            continue;
        }
        let tv = st.tensor(name).unwrap();
        assert_eq!(tv.dtype(), safetensors::Dtype::F32, "{name}");
        let shape = tv.shape().to_vec();
        let v: Vec<f32> = tv.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(v.len(), shape.iter().product::<usize>(), "{name}");
        map.insert(name.to_string(), (shape, v));
    }
    let l1 = load_spkconv(&mut map, "layer1.conv", 1152, 80, 5, 2, 1);
    let l1bn = take_bn(&mut map, "layer1.bn", 1152);
    let dilations = [2usize, 3, 4];
    let mut blocks = Vec::with_capacity(3);
    for (bi, &d) in dilations.iter().enumerate() {
        let lb = format!("layer{}", bi + 2);
        let c1 = load_spkconv(&mut map, &format!("{lb}.conv1.conv"), 1152, 1152, 1, 0, 1);
        let b1 = take_bn(&mut map, &format!("{lb}.conv1.bn"), 1152);
        let mut convs = Vec::with_capacity(7);
        let mut bns = Vec::with_capacity(7);
        for r in 0..7 {
            convs.push(load_spkconv(&mut map, &format!("{lb}.res2.convs.{r}"), 144, 144, 3, d, d));
            bns.push(take_bn(&mut map, &format!("{lb}.res2.bns.{r}"), 144));
        }
        let c2 = load_spkconv(&mut map, &format!("{lb}.conv2.conv"), 1152, 1152, 1, 0, 1);
        let b2 = take_bn(&mut map, &format!("{lb}.conv2.bn"), 1152);
        let (se1w, se1b) = pack_linear(&mut map, &format!("{lb}.se.linear1"), 128, 1152);
        let (se2w, se2b) = pack_linear(&mut map, &format!("{lb}.se.linear2"), 1152, 128);
        blocks.push(SpkSeBlock { c1, b1, res2: SpkRes2 { convs, bns }, c2, b2, se1w, se1b, se2w, se2b, channels: 1152 });
    }
    let cat_conv = load_spkconv(&mut map, "conv", 3456, 3456, 1, 0, 1);
    let (pool_w1, pool_b1) = pack_conv(&mut map, "pooling.linear1", 128, 10368, 1);
    let (pool_w2, pool_b2) = pack_conv(&mut map, "pooling.linear2", 3456, 128, 1);
    let final_bn = take_bn(&mut map, "bn", 6912);
    let (lin_w, lin_b) = pack_linear(&mut map, "linear", 256, 6912);
    assert!(map.is_empty(), "leftover tensors: {:?}", map.keys().collect::<Vec<_>>());
    SpeakerW {
        window: hann(WIN),
        mel_bank: mel_bank(),
        l1,
        l1bn,
        blocks: [blocks.remove(0), blocks.remove(0), blocks.remove(0)],
        cat_conv,
        pool_w1,
        pool_b1,
        pool_w2,
        pool_b2,
        final_bn,
        lin_w,
        lin_b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_matches_torch() {
        // torch.hann_window(400)[:4] = [0, 6.1690807e-05, 2.4673343e-04, ...]
        // torch computes in fp32 stepwise, we compute f64-then-cast:
        // agreement to ~1e-7 absolute (tap values themselves are ~1e-4).
        let w = hann(WIN);
        assert_eq!(w.len(), 400);
        assert!((w[0] - 0.0).abs() < 1e-9);
        assert!((w[1] - 6.1690807e-05).abs() < 1e-7, "{}", w[1]);
        assert!((w[280] - 0.6545085).abs() < 1e-6, "{}", w[280]);
        assert!((w[200] - 1.0).abs() < 1e-9);
        for v in w.iter() {
            assert!((0.0..=1.0).contains(v), "{v}");
        }
    }

    #[test]
    fn fft_delta_and_sine() {        // delta at 0 -> all bins exactly 1 (unnormalized forward DFT).
        let mut re = vec![0.0f32; 512];
        let mut im = vec![0.0f32; 512];
        re[0] = 1.0;
        fft(&mut re, &mut im);
        for k in 0..512 {
            assert!((re[k] - 1.0).abs() < 1e-4, "bin {k}: re={}", re[k]);
            assert!(im[k].abs() < 1e-4, "bin {k}: im={}", im[k]);
        }
        // cosine at bin 8 -> peaks at 8 and 504 with magnitude 256 each.
        let mut re = vec![0.0f32; 512];
        let mut im = vec![0.0f32; 512];
        for n in 0..512 {
            re[n] = (2.0 * std::f64::consts::PI * 8.0 * n as f64 / 512.0).cos() as f32;
        }
        fft(&mut re, &mut im);
        for k in 0..512 {
            let m = (re[k] * re[k] + im[k] * im[k]).sqrt();
            let want = if k == 8 || k == 504 { 256.0 } else { 0.0 };
            assert!((m - want).abs() < 0.5, "bin {k}: mag={m}");
        }
    }

    #[test]
    fn fft_shifted_delta() {        // delta at 336 (the STFT impulse position) -> |X| = 1 everywhere.
        let mut re = vec![0.0f32; 512];
        let mut im = vec![0.0f32; 512];
        re[336] = 1.0;
        fft(&mut re, &mut im);
        for k in 0..512 {
            let p = re[k] * re[k] + im[k] * im[k];
            assert!((p - 1.0).abs() < 1e-3, "bin {k}: power={p}");
        }
    }

    #[test]
    fn impulse_response_matches_torch() {
        // torch: x[400]=1 (800 zeros), center stft -> frames 2,3 carry
        // w[280]^2 = 0.4284 flat across all bins (verified in torch).
        let window = hann(WIN);
        let mut x = vec![0.0f32; 800];
        x[400] = 1.0;
        let pad = N_FFT / 2;
        let mut pw = vec![0.0f32; 800 + 2 * pad];
        for i in 0..pad {
            pw[i] = x[pad - i];
            pw[800 + pad + i] = x[800 - 2 - i];
        }
        pw[pad..pad + 800].copy_from_slice(&x);
        let woff = (N_FFT - WIN) / 2;
        for f in [2usize, 3] {
            let base = f * HOP;
            let mut re = vec![0.0f32; N_FFT];
            let mut im = vec![0.0f32; N_FFT];
            for n in 0..WIN {
                re[woff + n] = pw[base + woff + n] * window[n];
            }
            fft(&mut re, &mut im);
            for k in 0..N_FREQ {
                let p = re[k] * re[k] + im[k] * im[k];
                assert!((p - 0.4284).abs() < 1e-3, "frame {f} bin {k}: power={p}");
            }
        }
    }
}
