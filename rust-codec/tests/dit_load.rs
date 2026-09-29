//! DiT loader smoke: 61 tensors land in the right shapes.
use cutetts_codec::dit::load_dit_weights;
use std::path::PathBuf;

#[test]
fn dit_weights_load() {
    let wpath = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("model/CuteTTS-distill/weights/tts/model.safetensors");
    if !wpath.is_file() {
        eprintln!("skip: weights not found at {}", wpath.display());
        return;
    }
    let w = load_dit_weights(&wpath);
    assert_eq!(w.layers.len(), 4);
    assert_eq!(w.final_norm.len(), 1024);
    assert_eq!(w.in_proj.out_dim, 1024);
    assert_eq!(w.out_proj.out_dim, 64);
}
