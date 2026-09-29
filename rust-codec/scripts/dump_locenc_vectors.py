"""Dump LocEnc fixtures: torch (diffusion_head.AudioLocEnc) -> npy.

Production runs bf16 (model.py:92 casts locenc to backbone dtype), so dump
BOTH: fp32 reference (tight gate, proves implementation) and bf16 production
output (measures dtype noise floor for the later parity decision).
Path covered: embed_acoustic_latents = locenc_to_lm_proj(locenc(x)).

Shapes: A [1,8,64] (3D), B [1,4,2,64] (4D), C [1,1,2,64] (per-step patch).
Also writes locenc_manifest.json (24 tensors).
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

from cutetts.modeling.diffusion_head import AudioLocEnc  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

WEIGHTS = ROOT / "model/CuteTTS-distill/weights/tts/model.safetensors"


def save(name, t):
    t = np.ascontiguousarray(t.detach().cpu().float().numpy(), dtype=np.float32)
    np.save(OUT / f"{name}.npy", t)
    print(f"{name}: shape={list(t.shape)} maxabs={np.abs(t).max():.4f}")


def main() -> None:
    sd = load_file(WEIGHTS)
    loc_sd = {k[len("locenc."):]: v for k, v in sd.items() if k.startswith("locenc.")}
    proj_sd = {k[len("locenc_to_lm_proj."):]: v for k, v in sd.items() if k.startswith("locenc_to_lm_proj.")}
    print(f"locenc: {len(loc_sd)} tensors, proj: {len(proj_sd)} tensors")
    with open(OUT.parent / "locenc_manifest.json", "w") as f:
        json.dump(
            {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in sorted({**{f"locenc.{k}": v for k, v in loc_sd.items()}, **{f"locenc_to_lm_proj.{k}": v for k, v in proj_sd.items()}}.items())},
            f, indent=1,
        )

    def build(dtype):
        m = AudioLocEnc(input_dim=64, hidden_dim=1024, ffn_dim=4096, num_layers=2,
                        num_heads=16, num_kv_heads=2, patch_size=2,
                        rms_norm_eps=1e-6, use_rope=True, rope_theta=10000.0).to(dtype)
        missing, unexpected = m.load_state_dict(
            {k: v.to(dtype) for k, v in loc_sd.items()}, strict=True)
        assert not missing and not unexpected, (missing, unexpected)
        proj = torch.nn.Linear(1024, 1024).to(dtype)
        missing, unexpected = proj.load_state_dict(
            {k: v.to(dtype) for k, v in proj_sd.items()}, strict=True)
        assert not missing and not unexpected, (missing, unexpected)
        return m.eval(), proj.eval()

    m_bf16, p_bf16 = build(torch.bfloat16)
    m_fp32, p_fp32 = build(torch.float32)

    cases = {
        "b": torch.randn(1, 4, 2, 64),
        "c": torch.randn(1, 1, 2, 64),
        "d": torch.randn(2, 3, 2, 64),
    }
    with torch.no_grad():
        for tag, x in cases.items():
            save(f"loc_{tag}_in", x)
            out32 = p_fp32(m_fp32(x.float()))
            out16 = p_bf16(m_bf16(x.to(torch.bfloat16)))
            save(f"loc_{tag}_fp32", out32)
            save(f"loc_{tag}_bf16", out16)
            gap = (out32 - out16.float()).abs().max().item()
            print(f"loc_{tag}: bf16-vs-fp32 gap={gap:.2e}")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
