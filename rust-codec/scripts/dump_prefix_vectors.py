"""Dump prefix fixtures: input_ids + prefix embeds for tokenizer/prefix tests.

Texts: e2e text (cross-check e2e_prefix), Chinese, trickies (literal
<|im_start|> inside user text), empty-ish short.
Also verifies: no BOS added, <|endofprompt|> splits to id 16384.
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

TEXTS = {
    "e2e": "Hello world, this is a test.",
    "zh": "你好，这是中文测试。",
    "tricky": "Say <|im_start|> out loud.",
    "short": "Hi.",
}


def main() -> None:
    model = CuteTTS.from_pretrained(ROOT / "model/CuteTTS-distill", device="cpu")
    proc = model.runtime.processor
    tok = proc.tokenizer
    for tag, text in TEXTS.items():
        prompt = proc._text_only_prompt(text)
        ids = tok.encode(prompt)
        print(f"{tag}: prompt={prompt!r}")
        print(f"  ids[:8]={ids[:8]} ids[-4:]={ids[-4:]} len={len(ids)}")
        assert ids[-1] == 16384, "must end with <|endofprompt|>"
        np.save(OUT / f"tok_{tag}_ids.npy", np.array(ids, dtype=np.int64))
        with open(OUT / f"tok_{tag}_prompt.txt", "w") as f:
            f.write(prompt)
    # prefix embeds for e2e text (must equal e2e_prefix fixture)
    from cutetts.inference.conditioning import build_guidance_plan, build_prefix_segment
    plan = build_guidance_plan("tts", "nocfg", 2.0)
    seg = build_prefix_segment(proc, plan.conditional, target_text=TEXTS["e2e"])
    print("segment: total_length =", seg.total_length, "speaker_mask =", seg.speaker_linear_mask)
    with torch.no_grad():
        embeds, _, _ = model.runtime.model.prepare_input_embeds(seg, lm_speaker_embedding=None)
    prev = np.load(OUT / "e2e_prefix.npy")
    cur = embeds.detach().cpu().float().numpy()
    print(f"prefix embeds: shape={list(cur.shape)} vs e2e_prefix {list(prev.shape)} maxdiff={np.abs(cur - prev).max():.2e}")
    np.save(OUT / "tok_e2e_prefix.npy", np.ascontiguousarray(cur, dtype=np.float32))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
