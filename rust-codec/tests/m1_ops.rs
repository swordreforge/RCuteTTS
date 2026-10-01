//! M1: differential tests vs torch (`scripts/dump_m1_fixtures.py`).
//! Gate: max abs err < 1e-4 (f32 summation order differs from torch).

use cutetts_codec::{conv, snake};
use ndarray_npy::read_npy;
use std::path::PathBuf;

fn td(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata").join(name)
}

fn load2(name: &str) -> (Vec<f32>, usize, usize) {
    let a: ndarray::Array2<f32> = read_npy(td(name)).unwrap();
    let (c, t) = (a.shape()[0], a.shape()[1]);
    (a.into_raw_vec_and_offset().0, c, t)
}

fn load1(name: &str) -> Vec<f32> {
    let a: ndarray::Array1<f32> = read_npy(td(name)).unwrap();
    a.to_vec()
}

fn load3(name: &str) -> (Vec<f32>, usize, usize, usize) {
    let a: ndarray::Array3<f32> = read_npy(td(name)).unwrap();
    let s = a.shape().to_vec();
    (a.into_raw_vec_and_offset().0, s[0], s[1], s[2])
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn conv_dilated_d3() {
    let (x, ci, t) = load2("m1_conv_d3_in.npy");
    let (w, co, cig, k) = load3("m1_conv_d3_w.npy");
    let b = load1("m1_conv_d3_b.npy");
    let (ref_out, _, _) = load2("m1_conv_d3_out.npy");
    assert_eq!((ci, co, cig, k, t), (6, 6, 6, 7, 11));
    let y = conv::causal_conv1d(&x, ci, t, &w, &b, co, k, 3, 1, 9);
    let e = max_err(&y, &ref_out);
    assert!(e < 1e-4, "max_err={e}");
}

#[test]
fn conv_depthwise_d9() {
    let (x, ci, t) = load2("m1_conv_dw9_in.npy");
    let (w, co, cig, k) = load3("m1_conv_dw9_w.npy");
    let b = load1("m1_conv_dw9_b.npy");
    let (ref_out, _, _) = load2("m1_conv_dw9_out.npy");
    assert_eq!((ci, co, cig, k, t), (8, 8, 1, 7, 13));
    let y = conv::causal_conv1d(&x, ci, t, &w, &b, co, k, 9, 8, 27);
    let e = max_err(&y, &ref_out);
    assert!(e < 1e-4, "max_err={e}");
}

#[test]
fn transpose_even_s16() {
    let (x, ci, t) = load2("m1_trans_s16_in.npy");
    let (w, wi, co, k) = load3("m1_trans_s16_w.npy");
    let b = load1("m1_trans_s16_b.npy");
    let (ref_out, _, _) = load2("m1_trans_s16_out.npy");
    assert_eq!((ci, wi, co, k, t), (8, 8, 4, 32, 5));
    let y = conv::causal_transpose_conv1d(&x, ci, t, &w, &b, co, k, 16, 1, 8, 0);
    let e = max_err(&y, &ref_out);
    assert!(e < 1e-4, "max_err={e}");
}

#[test]
fn transpose_odd_s3() {
    let (x, ci, t) = load2("m1_trans_s3_in.npy");
    let (w, wi, co, k) = load3("m1_trans_s3_w.npy");
    let b = load1("m1_trans_s3_b.npy");
    let (ref_out, _, _) = load2("m1_trans_s3_out.npy");
    assert_eq!((ci, wi, co, k, t), (6, 6, 4, 6, 7));
    let y = conv::causal_transpose_conv1d(&x, ci, t, &w, &b, co, k, 3, 1, 2, 1);
    let e = max_err(&y, &ref_out);
    assert!(e < 1e-4, "max_err={e}");
}

#[test]
fn residual_unit_d3() {
    // block = snake -> conv(k7,d3) -> snake -> conv(k1) -> +residual
    let (x, c, t) = load2("m1_resunit_d3_in.npy");
    let (w0, _, _, _) = load3("m1_resunit_d3_w0.npy");
    let b0 = load1("m1_resunit_d3_b0.npy");
    let (w1, _, _, _) = load3("m1_resunit_d3_w1.npy");
    let b1 = load1("m1_resunit_d3_b1.npy");
    let (ref_out, _, _) = load2("m1_resunit_d3_out.npy");
    assert_eq!((c, t), (8, 11));
    let mut h = x.clone();
    let alpha = vec![1.0f32; c]; // fresh Snake1d inits alpha=1; dump used init model
    snake::snake1d(&mut h, c, t, &alpha);
    h = conv::causal_conv1d(&h, c, t, &w0, &b0, c, 7, 3, 1, 9);
    snake::snake1d(&mut h, c, t, &alpha);
    h = conv::causal_conv1d(&h, c, t, &w1, &b1, c, 1, 1, 1, 0);
    let y: Vec<f32> = x.iter().zip(h.iter()).map(|(a, b)| a + b).collect();
    let e = max_err(&y, &ref_out);
    assert!(e < 1e-4, "max_err={e}");
}

#[test]
fn snake_matches() {
    let (mut x, c, t) = load2("m1_snake_in.npy");
    let alpha = load1("m1_snake_alpha.npy");
    let (ref_out, _, _) = load2("m1_snake_out.npy");
    snake::snake1d(&mut x, c, t, &alpha);
    let e = max_err(&x, &ref_out);
    assert!(e < 1e-6, "max_err={e}");
}
