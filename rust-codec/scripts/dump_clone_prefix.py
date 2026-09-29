"""Dump clone-prefix intermediates: piece ids, reference features, final prefix.

Reference path: default_reference.wav (24k) -> VAE-encode -> [46,64] frames
-> speech segment [1,23,2,64] -> fused 5-segment prefix (len 71).
Saves ids for the 3 text pieces + reference_features + checks final prefix.
"""
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent.parent / "testdata"
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cutetts import CuteTTS  # noqa: E402
from cutetts.inference.conditioning import build_guidance_plan, build_prefix_segment  # noqa: E402
from cutetts.runtime import prepare_reference_audio  # noqa: E402

TEXT = "Hello world, this is a test."
REF = ROOT / "assets/default_reference.wav"


def save(name, t):
    t = np.ascontiguousarray(t.detach().cpu().float().numpy(), dtype=np.float32)
    np.save(OUT / f"{name}.npy", t)
    print(f"{name}: shape={list(t.shape)} maxabs={np.abs(t).max():.4f}")


def main() -> None:
    model = CuteTTS.from_pretrained(ROOT / "model/CuteTTS-distill", device="cpu")
    proc = model.runtime.processor
    tok = proc.tokenizer
    p1 = "Transform the text into speech output, utilizing the distinct voice of the provided speech sample.\nvoice reference:\n<|im_start|>"
    p2 = "<|im_end|>\n<|im_start|>"
    p3 = f"<|im_end|>\ntext input:\n{TEXT}\n{proc.text_suffix_token}"
    for tag, p in [("cp1", p1), ("cp2", p2), ("cp3", p3)]:
        ids = tok.encode(p)
        print(f"{tag}: len={len(ids)} tail={ids[-3:]}")
        np.save(OUT / f"clone_{tag}_ids.npy", np.array(ids, dtype=np.int64))
        with open(OUT / f"clone_{tag}_prompt.txt", "w") as f:
            f.write(p)

    ref_wave, _ = prepare_reference_audio(REF, 24000, 16000)
    with torch.no_grad():
        feats = proc.acoustic_vae.encode(ref_wave.unsqueeze(0)).mode()[0]
    print("reference_features:", list(feats.shape))
    save("clone_ref_feats", feats)

    plan = build_guidance_plan("voice_clone", "nocfg", 2.0)
    seg = build_prefix_segment(proc, plan.conditional, target_text=TEXT, reference_features=feats)
    print("fused: total =", seg.total_length, "audio =", seg.audio_length)
    with torch.no_grad():
        embeds, _, _ = model.runtime.model.prepare_input_embeds(seg, lm_speaker_embedding=None)
    prev = np.load(OUT / "clone_prefix.npy")
    cur = embeds.detach().cpu().float().numpy()
    print(f"prefix: shape={list(cur.shape)} vs clone_prefix maxdiff={np.abs(cur - prev).max():.2e}")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
