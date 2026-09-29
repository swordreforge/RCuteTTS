"""Dump Qwen3-7L fixtures: torch (transformers Qwen3Model, sdpa) -> npy.

Production path (distill, single branch, generation.py):
  P:  prefill inputs_embeds [1,48,1024], pos 0..47, empty DynamicCache
  D0-2: chained decodes [1,1,1024], pos 48/49/50, cache carried.
attention_mask=None everywhere (as in production).
Dual dtype: fp32 (tight gate) + bf16 production (noise floor).
Also writes qwen_manifest.json (79 tensors).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent.parent / "testdata"
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from transformers import Qwen3Config, Qwen3Model  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

WEIGHTS = ROOT / "model/CuteTTS-distill/weights/tts/model.safetensors"


def save(name, t):
    t = np.ascontiguousarray(t.detach().cpu().float().numpy(), dtype=np.float32)
    np.save(OUT / f"{name}.npy", t)
    print(f"{name}: shape={list(t.shape)} maxabs={np.abs(t).max():.4f}")


def build(dtype):
    lc = json.load(open(ROOT / "model/CuteTTS-distill/config.json"))["architecture"]["lm_config"]
    lc = dict(lc)
    lc["num_hidden_layers"] = 7
    lc["attn_implementation"] = "sdpa"  # as runtime.py (no flash_attn here)
    model = Qwen3Model(Qwen3Config(**lc)).to(dtype)
    model.resize_token_embeddings(16385)  # as model.py (extended_vocab_size)
    sd = load_file(WEIGHTS)
    qwen_sd = {k[len("qwen_backbone."):]: v.to(dtype) for k, v in sd.items() if k.startswith("qwen_backbone.")}
    if dtype == torch.float32 and not hasattr(build, "_manifest_done"):
        with open(OUT.parent / "qwen_manifest.json", "w") as f:
            json.dump({k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in sorted(load_file(WEIGHTS).items()) if k.startswith("qwen_backbone.")}, f, indent=1)
        n = sum(v.numel() for v in qwen_sd.values())
        print(f"qwen: {len(qwen_sd)} tensors, {n / 1e6:.1f}M params")
        build._manifest_done = True
    missing, unexpected = model.load_state_dict(qwen_sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return model.eval()


def run_one(model, embeds, pos, past):
    out = model(inputs_embeds=embeds, attention_mask=None,
                position_ids=pos, past_key_values=past, use_cache=True,
                return_dict=True)
    return out.last_hidden_state, out.past_key_values


def main() -> None:
    torch.manual_seed(400)
    pre = torch.randn(1, 48, 1024)
    dec_ins = [torch.randn(1, 1, 1024) for _ in range(3)]
    for dtype, tag in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
        model = build(dtype)
        with torch.no_grad():
            past = DynamicCache()
            h, past = run_one(model, pre.to(dtype),
                              torch.arange(48).reshape(1, -1), past)
            save(f"qwen_p_{tag}", pre)
            save(f"qwen_p_out_{tag}", h)
            for i, dins in enumerate(dec_ins):
                h, past = run_one(model, dins.to(dtype),
                                  torch.tensor([[48 + i]]), past)
                save(f"qwen_d{i}_{tag}", dins)
                save(f"qwen_d{i}_out_{tag}", h)
    # production bf16 vs fp32 gap on the last decode
    a = np.load(OUT / "qwen_d2_out_fp32.npy")
    b = np.load(OUT / "qwen_d2_out_bf16.npy")
    print(f"qwen decode bf16-vs-fp32 gap={np.abs(a - b).max():.2e} (out maxabs {np.abs(a).max():.2f})")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
