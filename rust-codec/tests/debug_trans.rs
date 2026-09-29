//! Debug: old scatter transpose vs new GEMM-decomposition transpose.
use cutetts_codec::conv::causal_transpose_conv1d;
use cutetts_codec::gemm::{pack_a, sgemm_bias};
use ndarray_npy::read_npy;
use std::path::PathBuf;

fn td(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata").join(name)
}

#[test]
fn transpose_gemm_matches_scatter() {
    // m1 fixture: W [I=8, O=4, K=32], s=16, pad=8, out_pad=0
    let xin: ndarray::Array2<f32> = read_npy(td("m1_trans_s16_in.npy")).unwrap();
    let w3: ndarray::Array3<f32> = read_npy(td("m1_trans_s16_w.npy")).unwrap();
    let b1: ndarray::Array1<f32> = read_npy(td("m1_trans_s16_b.npy")).unwrap();
    let (ci, t) = (xin.shape()[0], xin.shape()[1]);
    let (wi, co, k) = (w3.shape()[0], w3.shape()[1], w3.shape()[2]);
    assert_eq!((ci, wi, co, k, t), (8, 8, 4, 32, 5));
    let x = xin.into_raw_vec();
    let wflat = w3.into_raw_vec();
    let b = b1.to_vec();

    let ref_out =
        causal_transpose_conv1d(&x, ci, t, &wflat, &b, co, k, 16, 1, 8, 0);

    // new path: repack + sgemm + interleave (mirror of weights.rs/decode.rs)
    let (s, o, i_dim) = (16usize, co, ci);
    let mut stacked = vec![0.0f32; 2 * s * o * i_dim];
    for j in 0..s {
        for oo in 0..o {
            for c in 0..i_dim {
                stacked[(j * o + oo) * i_dim + c] = wflat[(c * o + oo) * 2 * s + j];
                stacked[((s + j) * o + oo) * i_dim + c] =
                    wflat[(c * o + oo) * 2 * s + s + j];
            }
        }
    }
    let mut bp = vec![0.0f32; 2 * s * o];
    for r in 0..s {
        bp[r * o..(r + 1) * o].copy_from_slice(&b);
    }
    let g = pack_a(&stacked, 2 * s * o, i_dim);
    let mut bp_pad = vec![0.0f32; g.rows];
    bp_pad[..bp.len()].copy_from_slice(&bp);
    let mut y = vec![0.0f32; g.rows * t];
    sgemm_bias(&g, &x, t, &bp_pad, &mut y, 1);
    let mut out = vec![0.0f32; o * t * s];
    for oo in 0..o {
        for ii in 0..t {
            for j in 0..s {
                let y0 = y[(j * o + oo) * t + ii];
                let y1 = if ii > 0 { y[((s + j) * o + oo) * t + ii - 1] } else { 0.0 };
                out[(oo * t + ii) * s + j] = y0 + y1;
            }
        }
    }
    assert_eq!(out.len(), ref_out.len());
    let e: f32 = out.iter().zip(ref_out.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    println!("transpose gemm-vs-scatter max_err={e:.2e}");

    // isolate: naive matmul (no sgemm) on the same stacked panels + same interleave
    let mut yn = vec![0.0f32; 2 * s * o * t];
    for r in 0..2 * s * o {
        let bb = if r < o { b[r] } else { 0.0 };
        for ii in 0..t {
            let mut acc = bb;
            for c in 0..i_dim {
                acc += stacked[r * i_dim + c] * x[c * t + ii];
            }
            yn[r * t + ii] = acc;
        }
    }
    let mut outn = vec![0.0f32; o * t * s];
    for oo in 0..o {
        for ii in 0..t {
            for j in 0..s {
                let y0 = yn[(j * o + oo) * t + ii];
                let y1 = if ii > 0 { yn[((s + j) * o + oo) * t + ii - 1] } else { 0.0 };
                outn[(oo * t + ii) * s + j] = y0 + y1;
            }
        }
    }
    let en: f32 = outn.iter().zip(ref_out.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    println!("transpose naive-panels-vs-scatter max_err={en:.2e}");
    let es: f32 = y.iter().zip(yn.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    println!("sgemm-vs-naive Y max_err={es:.2e}");
    assert!(e < 1e-4, "{e}");
}
