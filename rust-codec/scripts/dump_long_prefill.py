"""Teacher-forced long-prefill probe: real 1296-token tts prefix through fp32 Qwen.

Saves prefix embeds + fp32 prefill last-hidden-state for the Rust side
(examples/long_prefill.rs) to compare against. Isolates long-context
encoding correctness from chaotic free-run divergence.
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
from dump_qwen_vectors import build as build_qwen, run_one  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402


def save(name, t):
    t = np.ascontiguousarray(t.detach().cpu().float().numpy(), dtype=np.float32)
    np.save(OUT / f"{name}.npy", t)
    print(f"{name}: shape={list(t.shape)} maxabs={np.abs(t).max():.4f}", flush=True)


def main() -> None:
    text = open(ROOT / "rust-codec/result.txt").read().strip()
    model = CuteTTS.from_pretrained(ROOT / "model/CuteTTS-distill", device="cpu")
    # Real tts prompt path via the runtime segment machinery.
    from cutetts.inference.conditioning import build_guidance_plan, build_prefix_segment  # noqa: E402
    plan = build_guidance_plan("tts", "nocfg", 2.0)
    seg = build_prefix_segment(model.runtime.processor, plan.conditional, target_text=text)
    print("prefix total:", seg.total_length, flush=True)
    with torch.no_grad():
        embeds, _, _ = model.runtime.model.prepare_input_embeds(seg, lm_speaker_embedding=None)
    print("embeds:", list(embeds.shape), flush=True)
    save("long_pre", embeds[0])
    qwen = build_qwen(torch.float32)
    with torch.no_grad():
        h, _ = run_one(qwen, embeds.to(torch.float32),
                       torch.arange(embeds.shape[1]).reshape(1, -1), DynamicCache())
    save("long_pre_out", h[0])
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
