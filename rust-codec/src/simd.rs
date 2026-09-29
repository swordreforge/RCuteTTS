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

#[cfg(test)]
mod tests {
    use super::*;

    fn xrng(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
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
}
