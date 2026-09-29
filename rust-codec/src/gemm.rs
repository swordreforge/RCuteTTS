//! Minimal blocked f32 SGEMM for 1x1 convs and transposed-conv decomposition.
//!
//! `C = A*B + bias`, A is a static weight matrix pre-packed at load
//! (`pack_a`, MR=8 row interleave), B is row-major activations.
//! Compute kernel: explicit 8x8 register-blocked micro-kernel
//! ([`crate::simd::micro_8x8_fma`] / `micro_8x8_exact`) holding C in 8 ymm
//! accs across the full K loop — C traffic is O(1) per block instead of O(K).
//! Remainder cols (< 8) fall back to bias-fill + SAXPY.
//! Threading: 1D row chunks (small-T) or 2D row/time tiles (large-T).
//! No external deps (QORA-style: hand-rolled, thread::scope).
//! Hot inner loop uses explicit AVX2 kernels from [`crate::simd`]
//! (runtime-dispatched, portable binary).

use crate::simd::{resolve_saxpy, run_saxpy, SaxpyKind};

pub const MR: usize = 8;
const KC: usize = 256;
const NC: usize = 512;

pub struct PackedA {
    pub data: Vec<f32>,
    /// rows padded up to a multiple of MR
    pub rows: usize,
    pub cols: usize,
}

/// Pack row-major `a` (rows x cols) into `[rows/MR][cols][MR]` panels.
pub fn pack_a(a: &[f32], rows: usize, cols: usize) -> PackedA {
    assert_eq!(a.len(), rows * cols);
    let pr = rows.div_ceil(MR) * MR;
    let mut data = vec![0.0f32; pr * cols];
    for o in 0..rows {
        for i in 0..cols {
            data[(o / MR) * cols * MR + i * MR + (o % MR)] = a[o * cols + i];
        }
    }
    PackedA { data, rows: pr, cols }
}

/// `c[o, t] = bias[o] + sum_i A[o, i] * b[i, t]`, `b` row-major (cols x t).
/// `c` must hold `a.rows * t` floats (padded rows included).
/// 2D task grid (row-panels x time-tiles); each task owns a disjoint
/// ~16 x 2048 tile gathered/scattered through a contiguous temp buffer.
/// Kernel (exact/FMA) is resolved once here and threaded through.
pub fn sgemm_bias(a: &PackedA, b: &[f32], t: usize, bias: &[f32], c: &mut [f32], nth: usize) {
    sgemm_bias_with_kind(a, b, t, bias, c, nth, resolve_saxpy())
}

/// [`sgemm_bias`] with an explicit kernel (tests / debugging).
/// `SaxpyKind::{Fma, Exact}` select the 8x8 register micro-kernel;
/// `Scalar` keeps the old triple loop.
pub fn sgemm_bias_with_kind(
    a: &PackedA,
    b: &[f32],
    t: usize,
    bias: &[f32],
    c: &mut [f32],
    nth: usize,
    kind: SaxpyKind,
) {
    let (pr, k) = (a.rows, a.cols);
    assert_eq!(b.len(), k * t);
    assert_eq!(c.len(), pr * t);
    let ops = pr as u64 * t as u64 * k as u64;
    const ROW_STEP: usize = 2 * MR;
    const COL_STEP: usize = 2048;
    let rtasks = pr.div_ceil(ROW_STEP);
    let ctasks = t.div_ceil(COL_STEP);
    if nth <= 1 || ops < 200_000 || rtasks * ctasks <= 1 {
        return sgemm_range(a, b, t, bias, c, t, 0, pr, 0, t, kind);
    }
    // Small-T: 1D row chunks written directly (no temp tiles, no atomics).
    if ctasks == 1 {
        let panels = pr / MR;
        let per = (panels + nth - 1) / nth;
        std::thread::scope(|s| {
            for (pi, chunk) in c.chunks_mut(per * MR * t).enumerate() {
                let p0 = pi * per;
                let p1 = (p0 + per).min(panels);
                s.spawn(move || {
                    sgemm_range(a, b, t, bias, chunk, t, p0 * MR, p1 * MR, 0, t, kind);
                });
            }
        });
        return;
    }
    struct Task {
        r0: usize,
        r1: usize,
        t0: usize,
        t1: usize,
    }
    let mut tasks = Vec::with_capacity(rtasks * ctasks);
    for r in 0..rtasks {
        for cc in 0..ctasks {
            tasks.push(Task {
                r0: (r * ROW_STEP).min(pr),
                r1: ((r + 1) * ROW_STEP).min(pr),
                t0: (cc * COL_STEP).min(t),
                t1: ((cc + 1) * COL_STEP).min(t),
            });
        }
    }
    let c_ptr = c.as_mut_ptr() as usize;
    let next = std::sync::atomic::AtomicUsize::new(0);
    let next_ref = &next;
    let tasks_ref = &tasks;
    std::thread::scope(|s| {
        let workers = nth.min(tasks.len());
        for _ in 0..workers {
            s.spawn(move || loop {
                let idx = next_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if idx >= tasks_ref.len() {
                    break;
                }
                let task = &tasks_ref[idx];
                let (rn, tn) = (task.r1 - task.r0, task.t1 - task.t0);
                let mut tile = vec![0.0f32; rn * tn];
                sgemm_range(a, b, t, bias, &mut tile, tn, task.r0, task.r1, task.t0, task.t1, kind);
                unsafe {
                    let base = c_ptr as *mut f32;
                    for r in 0..rn {
                        let dst = base.add((task.r0 + r) * t + task.t0);
                        let src = tile.as_ptr().add(r * tn);
                        std::ptr::copy_nonoverlapping(src, dst, tn);
                    }
                }
            });
        }
    });
}

/// Compute output rows `[r0, r1)` x cols `[tc0, tc1)` into `c`,
/// a contiguous `(r1-r0)` x `(tc1-tc0)` row-major tile (`tn = tc1-tc0`).
///
/// Vector path (`Fma`/`Exact`): columns advance in 8-wide blocks, each
/// `(panel, block)` computed by one [`micro_8x8`] call whose 8 accumulators
/// stay in ymm regs across the full K loop — C traffic is O(1) per block
/// (bias init + single store) instead of O(K). Remainder cols (< 8) fall
/// back to bias-fill + SAXPY. `j0` (N-block) is the outer loop so a B
/// column-slice is reused across all row-panels while L2-resident.
fn sgemm_range(
    a: &PackedA,
    b: &[f32],
    t_full: usize,
    bias: &[f32],
    c: &mut [f32],
    tn: usize,
    r0: usize,
    r1: usize,
    tc0: usize,
    tc1: usize,
    kind: SaxpyKind,
) {
    let k = a.cols;
    if kind == SaxpyKind::Scalar {
        return sgemm_range_scalar(a, b, t_full, bias, c, tn, r0, r1, tc0, tc1);
    }
    let tcols = tc1 - tc0;
    let n_full = tcols / MR * MR;
    for j in (0..n_full).step_by(MR) {
        let jabs = tc0 + j;
        for p in (r0 / MR)..(r1 / MR) {
            let mut bb = [0.0f32; MR];
            for m in 0..MR {
                bb[m] = bias.get(p * MR + m).copied().unwrap_or(0.0);
            }
            let ap = &a.data[p * k * MR..p * k * MR + k * MR];
            let crow = &mut c[(p * MR - r0) * tn + j..];
            // SAFETY: ap holds exactly k*8 floats; crow has >= 8 cols
            // remaining in this row (j + 8 <= tn by loop bound).
            unsafe {
                match kind {
                    SaxpyKind::Fma => crate::simd::micro_8x8_fma(
                        ap.as_ptr(), b.as_ptr(), t_full, k, jabs,
                        bb.as_ptr(), crow.as_mut_ptr(), tn,
                    ),
                    _ => crate::simd::micro_8x8_exact(
                        ap.as_ptr(), b.as_ptr(), t_full, k, jabs,
                        bb.as_ptr(), crow.as_mut_ptr(), tn,
                    ),
                }
            }
        }
    }
    // Tail cols (< 8): bias fill + K-loop SAXPY (same as the old path).
    if n_full < tcols {
        let ker = kind;
        for p in (r0 / MR)..(r1 / MR) {
            for m in 0..MR {
                let bs = bias.get(p * MR + m).copied().unwrap_or(0.0);
                let row = &mut c[(p * MR - r0) * tn + m * tn + n_full
                    ..(p * MR - r0) * tn + (m + 1) * tn];
                row.fill(bs);
                for ii in 0..k {
                    let av = a.data[(p * k + ii) * MR + m];
                    let brow = &b[ii * t_full + tc0 + n_full..ii * t_full + tc1];
                    run_saxpy(ker, row, brow, av, tcols - n_full);
                }
            }
        }
    }
}

/// Scalar fallback for [`sgemm_range`] (non-x86 / no AVX2): the original
/// bias-fill + K-loop SAXPY nest.
fn sgemm_range_scalar(
    a: &PackedA,
    b: &[f32],
    t_full: usize,
    bias: &[f32],
    c: &mut [f32],
    tn: usize,
    r0: usize,
    r1: usize,
    tc0: usize,
    tc1: usize,
) {
    let k = a.cols;
    let ker = SaxpyKind::Scalar;
    for p in (r0 / MR)..(r1 / MR) {
        for m in 0..MR {
            let bs = bias.get(p * MR + m).copied().unwrap_or(0.0);
            let row = &mut c[(p * MR - r0) * tn + m * tn..(p * MR - r0) * tn + (m + 1) * tn];
            row.fill(bs);
        }
    }
    for t0 in (tc0..tc1).step_by(NC) {
        let t_end = (t0 + NC).min(tc1);
        let nw = t_end - t0;
        for i0 in (0..k).step_by(KC) {
            let kn = (i0 + KC).min(k);
            for p in (r0 / MR)..(r1 / MR) {
                let ap = &a.data[p * k * MR..];
                let co = (p * MR - r0) * tn;
                for ii in i0..kn {
                    let aoff = ii * MR;
                    let brow = &b[ii * t_full + t0..ii * t_full + t0 + nw];
                    for m in 0..MR {
                        let av = ap[aoff + m];
                        let crow = &mut c[co + m * tn + (t0 - tc0)..co + m * tn + (t0 - tc0) + nw];
                        run_saxpy(ker, crow, brow, av, nw);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn small_gemm_exact() {
        // 3x4 (padded to 8) * 4x5
        let a: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let b: Vec<f32> = (0..20).map(|v| (v as f32) * 0.25).collect();
        let pa = pack_a(&a, 3, 4);
        let mut c = vec![0.0f32; pa.rows * 5];
        sgemm_bias(&pa, &b, 5, &[1.0, 2.0, 3.0], &mut c, 1);
        for o in 0..3 {
            for tt in 0..5 {
                let mut e = (o + 1) as f32;
                for i in 0..4 {
                    e += a[o * 4 + i] * b[i * 5 + tt];
                }
                assert!((c[o * 5 + tt] - e).abs() < 1e-5, "{o},{tt}");
            }
        }
    }

    #[test]
    fn threaded_matches_serial() {
        let o = 64;
        let k = 128;
        let t = 300;
        let a: Vec<f32> = (0..o * k).map(|v| (v % 97) as f32 * 0.01).collect();
        let b: Vec<f32> = (0..k * t).map(|v| (v % 53) as f32 * 0.02).collect();
        let pa = pack_a(&a, o, k);
        let bias: Vec<f32> = (0..o).map(|v| v as f32 * 0.1).collect();
        let mut c1 = vec![0.0f32; pa.rows * t];
        let mut c8 = vec![0.0f32; pa.rows * t];
        sgemm_bias(&pa, &b, t, &bias, &mut c1, 1);
        sgemm_bias(&pa, &b, t, &bias, &mut c8, 8);
        for (x, y) in c1.iter().zip(c8.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }
}
