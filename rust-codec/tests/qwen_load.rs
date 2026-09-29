//! Qwen3 loader smoke: 79 tensors land in the right shapes.
use cutetts_codec::qwen::load_qwen_weights;
use std::path::PathBuf;

#[test]
fn qwen_weights_load() {
    let wpath = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("model/CuteTTS-distill/weights/tts/model.safetensors");
    if !wpath.is_file() {
        eprintln!("skip: weights not found at {}", wpath.display());
        return;
    }
    let t0 = std::time::Instant::now();
    let w = load_qwen_weights(&wpath);
    println!("qwen loaded in {:.2}s", t0.elapsed().as_secs_f32());
    assert_eq!(w.layers.len(), 7);
    assert_eq!(w.embed.len(), 16385 * 1024);
    assert_eq!(w.layers[0].q.out_dim, 2048);
    assert_eq!(w.layers[0].k.out_dim, 1024);
    assert_eq!(w.layers[0].qn.len(), 128);
    assert_eq!(w.final_norm.len(), 1024);
}
