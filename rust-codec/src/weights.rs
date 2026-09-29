//! Load + fuse AudioVAE decoder weights from safetensors.
//!
//! Source: `model/CuteTTS/weights/audio_vae/model.safetensors` (fp32).
//! Every conv carries `weight_g [O,1,1] + weight_v [O,I,K] + bias [O]`;
//! fused offline here to plain `w [O,I,K]` via [`crate::conv::fuse_weight_norm`].

use crate::conv::fuse_weight_norm;
use safetensors::SafeTensors;
use std::collections::HashMap;

pub struct ResUnitW {
    pub alpha0: Vec<f32>,
    pub dw_w: Vec<f32>,
    pub dw_b: Vec<f32>,
    pub alpha1: Vec<f32>,
    pub pw_w: Vec<f32>,
    pub pw_b: Vec<f32>,
    pub dim: usize,
}

pub struct StageW {
    pub alpha: Vec<f32>,
    pub trans_w: Vec<f32>,
    pub trans_b: Vec<f32>,
    pub in_c: usize,
    pub out_c: usize,
    pub stride: usize,
    pub res: [ResUnitW; 3],
}

pub struct DecoderW {
    pub front_dw_w: Vec<f32>,
    pub front_dw_b: Vec<f32>,
    pub front_pw_w: Vec<f32>,
    pub front_pw_b: Vec<f32>,
    pub stages: [StageW; 4],
    pub final_alpha: Vec<f32>,
    pub final_w: Vec<f32>,
    pub final_b: Vec<f32>,
}

fn flat(map: &HashMap<String, (Vec<usize>, Vec<f32>)>, name: &str) -> (Vec<usize>, Vec<f32>) {
    map.get(name)
        .unwrap_or_else(|| panic!("missing tensor {name}"))
        .clone()
}

fn fused(map: &HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str) -> (Vec<usize>, Vec<f32>) {
    let (sg, g) = flat(map, &format!("{prefix}.weight_g"));
    let (sv, v) = flat(map, &format!("{prefix}.weight_v"));
    assert_eq!(sg, vec![sv[0], 1, 1], "g shape for {prefix}");
    let (o, i, k) = (sv[0], sv[1], sv[2]);
    let w = fuse_weight_norm(&g, &v, o, i, k);
    (vec![o, i, k], w)
}

fn bias(map: &HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str) -> Vec<f32> {
    flat(map, &format!("{prefix}.bias")).1
}

fn alpha(map: &HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str) -> Vec<f32> {
    // stored [1, C, 1]
    let (s, v) = flat(map, &format!("{prefix}.alpha"));
    assert_eq!(s.len(), 3);
    assert_eq!(v.len(), s[1]);
    v
}

pub fn load_decoder_weights(path: &std::path::Path) -> DecoderW {
    let bytes = std::fs::read(path).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let mut map: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    for name in st.names() {
        let tv = st.tensor(name).unwrap();
        assert_eq!(tv.dtype(), safetensors::Dtype::F32, "{name}");
        let shape = tv.shape().to_vec();
        let raw = tv.data();
        assert_eq!(raw.len() % 4, 0);
        let v: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        assert_eq!(v.len(), shape.iter().product::<usize>(), "{name}");
        map.insert(name.to_string(), (shape, v));
    }
    let (s0, front_dw_w) = fused(&map, "decoder.model.0");
    assert_eq!(s0, vec![64, 1, 7]);
    let (s1, front_pw_w) = fused(&map, "decoder.model.1");
    assert_eq!(s1, vec![1536, 64, 1]);

    let dims = [(1536usize, 768usize, 16usize), (768, 384, 8), (384, 192, 5), (192, 96, 3)];
    let mut stages = Vec::with_capacity(4);
    for (si, (inc, outc, stride)) in dims.iter().enumerate() {
        let base = format!("decoder.model.{}.block", si + 2);
        let a = alpha(&map, &format!("{base}.0"));
        assert_eq!(a.len(), *inc);
        let (st, trans_w) = fused(&map, &format!("{base}.1"));
        assert_eq!(st, vec![*inc, *outc, 2 * stride], "transpose {base}.1");
        let trans_b = bias(&map, &format!("{base}.1"));
        let mut res = Vec::with_capacity(3);
        for ri in 2..=4 {
            let rb = format!("{base}.{ri}");
            let a0 = alpha(&map, &format!("{rb}.block.0"));
            let (sd, dw_w) = fused(&map, &format!("{rb}.block.1"));
            assert_eq!(sd, vec![*outc, 1, 7], "dw {rb}");
            let dw_b = bias(&map, &format!("{rb}.block.1"));
            let a1 = alpha(&map, &format!("{rb}.block.2"));
            let (sp, pw_w) = fused(&map, &format!("{rb}.block.3"));
            assert_eq!(sp, vec![*outc, *outc, 1], "pw {rb}");
            let pw_b = bias(&map, &format!("{rb}.block.3"));
            res.push(ResUnitW { alpha0: a0, dw_w, dw_b, alpha1: a1, pw_w, pw_b, dim: *outc });
        }
        stages.push(StageW {
            alpha: a,
            trans_w,
            trans_b,
            in_c: *inc,
            out_c: *outc,
            stride: *stride,
            res: [res.remove(0), res.remove(0), res.remove(0)],
        });
    }
    let fa = alpha(&map, "decoder.model.6");
    assert_eq!(fa.len(), 96);
    let (sf, final_w) = fused(&map, "decoder.model.7");
    assert_eq!(sf, vec![1, 96, 7]);
    DecoderW {
        front_dw_w,
        front_dw_b: bias(&map, "decoder.model.0"),
        front_pw_w,
        front_pw_b: bias(&map, "decoder.model.1"),
        stages: [stages.remove(0), stages.remove(0), stages.remove(0), stages.remove(0)],
        final_alpha: fa,
        final_w,
        final_b: bias(&map, "decoder.model.7"),
    }
}
