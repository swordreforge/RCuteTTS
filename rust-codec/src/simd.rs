//! Explicit AVX2 SAXPY kernels for the GEMM hot loop (QORA-style).
//!
//! Pattern mirrors `causal_range_body` in QORA `simd.rs`: 8-wide over time,
//! broadcast scalar, load acc, mul+add or FMA, store. Runtime dispatch via
//! `is_x86_feature_detected!`, so the binary stays portable (no
//! `-C target-cpu=native` required).
//!
//! - `saxpy_exact`: mul + add separately. Bit-identical to the scalar
//!   `crow[n] += av * brow[n]` loop (same op order, same roundings).
//! - `saxpy_fma`: single-rounding FMA, ~2x throughput on AVX2+FMA machines,
//!   ~1e-7 relative difference. Gated by `CUTETTS_FMA=0` (force exact).

#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2+FMA available at runtime (our target: Ultra 7 155H has both, no AVX512).
pub fn has_avx2_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Env gate: `CUTETTS_FMA=0` forces the bit-exact mul+add path (debugging).
pub fn fma_allowed() -> bool {
    std::env::var("CUTETTS_FMA").map(|v| v != "0").unwrap_or(true)
}

/// Resolved kernel selector. Call ONCE per GEMM (not per SAXPY):
/// `env::var` costs ~100ns, and a full decode issues tens of millions
/// of SAXPY calls.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SaxpyKind {
    Scalar,
    Exact,
    Fma,
}

/// Hoisted dispatch: feature detection + env lookup happen once here.
pub fn resolve_saxpy() -> SaxpyKind {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() && fma_allowed() {
            return SaxpyKind::Fma;
        }
        if has_avx2() {
            return SaxpyKind::Exact;
        }
    }
    SaxpyKind::Scalar
}

/// Run an already-resolved kernel (hot-loop entry point, no env lookup).
#[inline]
pub fn run_saxpy(kind: SaxpyKind, c: &mut [f32], b: &[f32], av: f32, n: usize) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        match kind {
            SaxpyKind::Fma => saxpy_fma_avx2(c.as_mut_ptr(), b.as_ptr(), av, n),
            SaxpyKind::Exact => saxpy_exact_avx2(c.as_mut_ptr(), b.as_ptr(), av, n),
            SaxpyKind::Scalar => saxpy_scalar(c, b, av, n),
        }
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = kind;
        saxpy_scalar(c, b, av, n)
    }
}

/// `c[n] += av * b[n]` for `n` lanes. Scalar fallback (also the oracle).
#[inline]
pub fn saxpy_scalar(c: &mut [f32], b: &[f32], av: f32, n: usize) {
    for i in 0..n {
        c[i] += av * b[i];
    }
}

/// Bit-exact AVX2 mirror of [`saxpy_scalar`]: mul then add, 8-wide.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn saxpy_exact_avx2(c: *mut f32, b: *const f32, av: f32, n: usize) {
    let wb = _mm256_set1_ps(av);
    let mut i = 0;
    while i + 8 <= n {
        let acc = _mm256_loadu_ps(c.add(i));
        let x = _mm256_loadu_ps(b.add(i));
        _mm256_storeu_ps(c.add(i), _mm256_add_ps(acc, _mm256_mul_ps(x, wb)));
        i += 8;
    }
    while i < n {
        *c.add(i) += av * *b.add(i);
        i += 1;
    }
}

/// FMA variant: single rounding, ~10-15% faster than mul+add on Intel
/// (4 uops vs 5 per 8 outputs), ~1e-7 relative difference.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn saxpy_fma_avx2(c: *mut f32, b: *const f32, av: f32, n: usize) {
    let wb = _mm256_set1_ps(av);
    let mut i = 0;
    while i + 8 <= n {
        let acc = _mm256_loadu_ps(c.add(i));
        let x = _mm256_loadu_ps(b.add(i));
        _mm256_storeu_ps(c.add(i), _mm256_fmadd_ps(x, wb, acc));
        i += 8;
    }
    while i < n {
        *c.add(i) += av * *b.add(i);
        i += 1;
    }
}

/// Dispatched SAXPY: FMA when allowed + available, else exact AVX2, else scalar.
#[inline]
pub fn saxpy(c: &mut [f32], b: &[f32], av: f32, n: usize) {
    // Cold-path convenience wrapper. Hot loops must hoist with
    // `resolve_saxpy()` + `run_saxpy()` instead (env lookup per call
    // costs ~100ns x tens of millions of calls per decode).
    run_saxpy(resolve_saxpy(), c, b, av, n)
}

// ============================================================
// 8x8 register-blocked micro-kernel (GotoBLAS-style).
// ============================================================
//
// Problem with the SAXPY loop order: every K iteration does
// load-C + load-B + FMA + store-C on the whole C tile, i.e. the
// tile round-trips through L1 K times (K <= 1536 here).
// The micro-kernel instead holds an 8-row x 8-col C block in 8 ymm
// accumulators across the FULL K loop: C traffic drops from O(K)
// to O(1) (init from bias + single store). A streams once
// (broadcasts), B streams once (8-wide loads).
//
// Layout contract (matches `pack_a`, MR=8):
// - `ap`: one A panel, K*8 contiguous floats, row ii at `ap + ii*8`.
// - `b`: full B matrix, K x `t_full` row-major; block cols `j0..j0+8`.
// - `bias8`: 8 bias floats for the 8 rows.
// - `c`: C tile row `m` starts at `c + m*c_stride`; store 8 floats at +0.
// - Op order per lane is bias, then ii=0..K ascending — identical to
//   the SAXPY path, so `exact` is bit-identical and `fma` differs only
//   by single-vs-double rounding (same as before, no new error source).

/// Scalar 8x8 micro-kernel (oracle + non-x86 fallback).
/// Lane (m, j) order: bias, then ii=0..K ascending — matches the vector
/// kernel, so `exact` must agree bitwise.
pub fn micro_8x8_scalar(
    ap: &[f32],
    b: &[f32],
    t_full: usize,
    k: usize,
    j0: usize,
    bias8: &[f32; 8],
    c: &mut [f32],
    c_stride: usize,
) {
    let mut acc = [[0.0f32; 8]; 8];
    for m in 0..8 {
        for j in 0..8 {
            acc[m][j] = bias8[m];
        }
    }
    for ii in 0..k {
        for m in 0..8 {
            let a = ap[ii * 8 + m];
            for j in 0..8 {
                acc[m][j] += a * b[ii * t_full + j0 + j];
            }
        }
    }
    for m in 0..8 {
        c[m * c_stride..m * c_stride + 8].copy_from_slice(&acc[m]);
    }
}

/// Shared 8x8 body (QORA `causal_range_body` pattern): `$op(bcast, bvec, acc)`
/// is mul+add for exact, fmadd for FMA. Accumulators stay in 8 ymm regs
/// across the full K loop — C is touched exactly twice (bias init, store).
#[cfg(target_arch = "x86_64")]
macro_rules! micro8_body {
    ($ap:expr, $b:expr, $t_full:expr, $k:expr, $j0:expr, $bias8:expr, $c:expr, $cs:expr, $op:expr) => {{
        let mut a0 = _mm256_set1_ps((*$bias8.add(0)));
        let mut a1 = _mm256_set1_ps((*$bias8.add(1)));
        let mut a2 = _mm256_set1_ps((*$bias8.add(2)));
        let mut a3 = _mm256_set1_ps((*$bias8.add(3)));
        let mut a4 = _mm256_set1_ps((*$bias8.add(4)));
        let mut a5 = _mm256_set1_ps((*$bias8.add(5)));
        let mut a6 = _mm256_set1_ps((*$bias8.add(6)));
        let mut a7 = _mm256_set1_ps((*$bias8.add(7)));
        let combine = $op;
        for ii in 0..$k {
            let bv = _mm256_loadu_ps($b.add(ii * $t_full + $j0));
            let bp = $ap.add(ii * 8);
            a0 = combine(_mm256_broadcast_ss(&*bp.add(0)), bv, a0);
            a1 = combine(_mm256_broadcast_ss(&*bp.add(1)), bv, a1);
            a2 = combine(_mm256_broadcast_ss(&*bp.add(2)), bv, a2);
            a3 = combine(_mm256_broadcast_ss(&*bp.add(3)), bv, a3);
            a4 = combine(_mm256_broadcast_ss(&*bp.add(4)), bv, a4);
            a5 = combine(_mm256_broadcast_ss(&*bp.add(5)), bv, a5);
            a6 = combine(_mm256_broadcast_ss(&*bp.add(6)), bv, a6);
            a7 = combine(_mm256_broadcast_ss(&*bp.add(7)), bv, a7);
        }
        _mm256_storeu_ps($c.add(0 * $cs), a0);
        _mm256_storeu_ps($c.add(1 * $cs), a1);
        _mm256_storeu_ps($c.add(2 * $cs), a2);
        _mm256_storeu_ps($c.add(3 * $cs), a3);
        _mm256_storeu_ps($c.add(4 * $cs), a4);
        _mm256_storeu_ps($c.add(5 * $cs), a5);
        _mm256_storeu_ps($c.add(6 * $cs), a6);
        _mm256_storeu_ps($c.add(7 * $cs), a7);
    }};
}

/// 8x8 micro-kernel, bit-exact vs [`micro_8x8_scalar`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn micro_8x8_exact(
    ap: *const f32,
    b: *const f32,
    t_full: usize,
    k: usize,
    j0: usize,
    bias8: *const f32,
    c: *mut f32,
    c_stride: usize,
) {
    micro8_body!(
        ap, b, t_full, k, j0, bias8, c, c_stride,
        |x: __m256, y: __m256, z: __m256| _mm256_add_ps(z, _mm256_mul_ps(x, y))
    );
}

/// 8x8 micro-kernel, FMA (single rounding; ~1 ulp vs exact).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn micro_8x8_fma(
    ap: *const f32,
    b: *const f32,
    t_full: usize,
    k: usize,
    j0: usize,
    bias8: *const f32,
    c: *mut f32,
    c_stride: usize,
) {
    micro8_body!(
        ap, b, t_full, k, j0, bias8, c, c_stride,
        |x: __m256, y: __m256, z: __m256| _mm256_fmadd_ps(x, y, z)
    );
}

// ============================================================
// bf16 -> f32 conversion (LM ticket: Qwen3-7L + LocEnc are bf16).
// ============================================================
//
// bf16 is exactly the top 16 bits of f32, so conversion is bit-preserving:
// `f32_bits = (bf16_bits as u32) << 16` — no rounding, no tolerance needed,
// NaN/Inf payloads preserved. AVX2 path: 128-bit u16 load -> cvtepu16 ->
// slli 16 -> bitcast store, 8 lanes per iteration.

/// Scalar bf16->f32 oracle (also the non-x86 fallback).
#[inline]
pub fn bf16_to_f32_scalar(dst: &mut [f32], src: &[u16]) {
    assert_eq!(dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = f32::from_bits((s as u32) << 16);
    }
}

/// AVX2 bf16->f32: bitwise identical to [`bf16_to_f32_scalar`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn bf16_to_f32_avx2(dst: *mut f32, src: *const u16, n: usize) {
    let mut i = 0;
    while i + 8 <= n {
        let v16 = _mm_loadu_si128(src.add(i) as *const __m128i);
        let v32 = _mm256_cvtepu16_epi32(v16);
        let shifted = _mm256_slli_epi32::<16>(v32);
        _mm256_storeu_ps(dst.add(i), _mm256_castsi256_ps(shifted));
        i += 8;
    }
    while i < n {
        *dst.add(i) = f32::from_bits((*src.add(i) as u32) << 16);
        i += 1;
    }
}

/// Dispatched bf16->f32 (exact on all paths).
#[inline]
pub fn bf16_to_f32(dst: &mut [f32], src: &[u16]) {
    assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            return unsafe { bf16_to_f32_avx2(dst.as_mut_ptr(), src.as_ptr(), src.len()) };
        }
    }
    bf16_to_f32_scalar(dst, src)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xrng(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn rvec(st: &mut u64, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| (xrng(st) % 2000) as f32 * 0.001 * scale - scale * 0.5).collect()
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn micro_exact_matches_scalar_bitwise() {
        if !has_avx2() {
            eprintln!("no AVX2, skipping");
            return;
        }
        let mut st = 99u64;
        // (k, t_full, j0, c_stride): cover k < / == / > KC, strided C, j0 != 0
        for &(k, t_full, j0, cs) in &[
            (1usize, 8, 0, 8),
            (7, 16, 0, 16),
            (64, 24, 8, 24),
            (300, 120, 40, 120),
            (1536, 120, 0, 120),
        ] {
            let ap = rvec(&mut st, k * 8, 2.0);
            let b = rvec(&mut st, k * t_full, 2.0);
            let bias: [f32; 8] = core::array::from_fn(|_| (xrng(&mut st) % 1000) as f32 * 0.01);
            let mut c1 = vec![9.0f32; 8 * cs];
            let mut c2 = c1.clone();
            micro_8x8_scalar(&ap, &b, t_full, k, j0, &bias, &mut c1, cs);
            unsafe { micro_8x8_exact(ap.as_ptr(), b.as_ptr(), t_full, k, j0, bias.as_ptr(), c2.as_mut_ptr(), cs) };
            assert_eq!(c1.len(), c2.len());
            for (i, (a, b)) in c1.iter().zip(c2.iter()).enumerate() {
                assert!(a.to_bits() == b.to_bits(), "k={k} t={t_full} j0={j0} cs={cs} [{i}]: {a} vs {b}");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn micro_fma_within_tolerance() {        if !has_avx2_fma() {
            eprintln!("no AVX2+FMA, skipping");
            return;
        }
        let mut st = 123u64;
        let (k, t_full, j0, cs) = (700usize, 120, 16, 120);
        let ap = rvec(&mut st, k * 8, 2.0);
        let b = rvec(&mut st, k * t_full, 2.0);
        let bias: [f32; 8] = core::array::from_fn(|_| (xrng(&mut st) % 1000) as f32 * 0.01);
        let mut c1 = vec![0.0f32; 8 * cs];
        let mut c2 = vec![0.0f32; 8 * cs];
        micro_8x8_scalar(&ap, &b, t_full, k, j0, &bias, &mut c1, cs);
        unsafe { micro_8x8_fma(ap.as_ptr(), b.as_ptr(), t_full, k, j0, bias.as_ptr(), c2.as_mut_ptr(), cs) };
        let worst = c1.iter().zip(c2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        let scale = c1.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-6);
        assert!(worst / scale < 1e-6, "worst={worst:.2e} scale={scale:.2e}");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn exact_matches_scalar_bitwise() {
        if !has_avx2() {
            eprintln!("no AVX2, skipping");
            return;
        }
        let mut st = 42u64;
        for &n in &[1usize, 7, 8, 9, 15, 16, 24, 120, 511, 512] {
            let mut c1 = vec![0.0f32; n];
            let mut c2 = vec![0.0f32; n];
            let b: Vec<f32> = (0..n).map(|_| (xrng(&mut st) % 1000) as f32 * 0.01 - 5.0).collect();
            for i in 0..n {
                c1[i] = (xrng(&mut st) % 1000) as f32 * 0.01;
                c2[i] = c1[i];
            }
            let av = 1.2345;
            saxpy_scalar(&mut c1, &b, av, n);
            unsafe { saxpy_exact_avx2(c2.as_mut_ptr(), b.as_ptr(), av, n) };
            for (i, (a, b)) in c1.iter().zip(c2.iter()).enumerate() {
                assert!(a.to_bits() == b.to_bits(), "n={n} [{i}]: {a} vs {b}");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn fma_within_tolerance() {
        if !has_avx2_fma() {
            eprintln!("no AVX2+FMA, skipping");
            return;
        }
        let mut st = 7u64;
        let n = 512;
        let mut c1 = vec![0.0f32; n];
        let mut c2 = vec![0.0f32; n];
        let b: Vec<f32> = (0..n).map(|_| (xrng(&mut st) % 2000) as f32 * 0.001 - 1.0).collect();
        for i in 0..n {
            c1[i] = (xrng(&mut st) % 2000) as f32 * 0.001;
            c2[i] = c1[i];
        }
        let av = 0.987;
        saxpy_scalar(&mut c1, &b, av, n);
        unsafe { saxpy_fma_avx2(c2.as_mut_ptr(), b.as_ptr(), av, n) };
        // Single FMA vs mul+add: <= ~1 ulp of |av*b| per lane (~1e-7 at O(1)).
        // Relative-to-result can spike where the result is near zero
        // (cancellation), so gate at 1e-5; the M2 waveform gate (1e-3 abs)
        // still has 100x margin.
        let max_rel = c1.iter().zip(c2.iter()).map(|(a, b)| (a - b).abs() / a.abs().max(1e-6)).fold(0.0, f32::max);
        assert!(max_rel < 1e-5, "max_rel={max_rel}");
    }

    #[test]
    fn bf16_to_f32_bitwise() {
        // bit-preserving: f32_bits == (bf16_bits as u32) << 16 for ALL patterns
        // (normals, subnormals, zeros, inf, nan) — no tolerance needed.
        let mut st = 555u64;
        // exhaustive lows + random + specials
        let mut src: Vec<u16> = (0..=0xFFFFu32).step_by(997).map(|v| v as u16).collect();
        src.extend((0..4096).map(|_| (xrng(&mut st) & 0xFFFF) as u16));
        src.extend([0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x0001, 0x7F7F, 0x3C00]);
        let mut d1 = vec![0.0f32; src.len()];
        let mut d2 = vec![0.0f32; src.len()];
        bf16_to_f32_scalar(&mut d1, &src);
        bf16_to_f32(&mut d2, &src);
        for (i, ((a, b), &s)) in d1.iter().zip(d2.iter()).zip(src.iter()).enumerate() {
            assert!(a.to_bits() == b.to_bits(), "[{i}] s={s:#06x}: {a} vs {b}");
            assert_eq!(a.to_bits(), (s as u32) << 16, "[{i}] shift identity");
        }
        // cross-check against the `half` crate as an independent oracle.
        // NOTE: restricted to non-NaN: `half` canonicalizes NaN payloads
        // while shift (like torch/CUDA __bfloat162float) preserves bits.
        // Weights are finite, so this never matters for loading.
        for &s in &src {
            if s & 0x7F80 == 0x7F80 && s & 0x007F != 0 {
                continue; // NaN: payload handling differs by design
            }
            let h = half::bf16::from_bits(s).to_f32().to_bits();
            assert_eq!(h, (s as u32) << 16, "half crate disagrees on {s:#06x}");
        }
    }
}
