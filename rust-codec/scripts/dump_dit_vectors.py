"""Dump DiT head fixtures: torch (diffusion_head.py) -> npy for Rust tests.

Covers the distill inference path only:
  A. unit _predict (t=0.25, dt=0.25, w=2.0), 3 seeds
  B. full sample() (steps=4, cfg=0, w=2.0, sway=0), 2 seeds (randn seeded)
Also writes dit_manifest.json (shapes/dtypes of head.*).

Run: source ../.venv/bin/activate && python dump_dit_vectors.py
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

from cutetts.modeling.diffusion_head import AudioDiTHead  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

WEIGHTS = ROOT / "model/CuteTTS-distill/weights/tts/model.safetensors"


def build_head() -> AudioDiTHead:
    head = AudioDiTHead(
        latent_dim=64,
        cond_dim=1024,
        hidden_dim=1024,
        ffn_dim=4096,
        num_layers=4,
        num_heads=16,
        num_kv_heads=2,
        patch_size=2,
        rms_norm_eps=1e-6,
        use_rope=True,
        rope_theta=10000.0,
        speaker_adaln_zero_enabled=True,
        speaker_embedding_dim=256,
        cfg_strength_embedding_enabled=True,
        cfg_strength_max=5.0,
        step_size_embedding_enabled=True,
        step_schedule="uniform",
    )
    sd = load_file(WEIGHTS)
    head_sd = {k[len("head."):]: v for k, v in sd.items() if k.startswith("head.")}
    missing, unexpected = head.load_state_dict(head_sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    # manifest (M0 routine)
    manifest = {
        k: {"shape": list(v.shape), "dtype": str(v.dtype)}
        for k, v in sorted(head_sd.items())
    }
    with open(OUT.parent / "dit_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    n = sum(v.numel() for v in head_sd.values())
    print(f"head: {len(head_sd)} tensors, {n / 1e6:.1f}M params")
    return head.eval()


def save(name, t):
    t = np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32)
    np.save(OUT / f"{name}.npy", t)
    print(f"{name}: shape={list(t.shape)} maxabs={np.abs(t).max():.4f}")


def main() -> None:
    head = build_head()
    print("mu_proj is", type(head.mu_proj).__name__)
    with torch.no_grad():
        # A. unit _predict
        for i in range(3):
            torch.manual_seed(200 + i)
            x = torch.randn(1, 2, 64)
            t = torch.full((1,), 0.25)
            z = torch.randn(1, 1024)
            cond = torch.randn(1, 2, 64)
            spk = torch.randn(1, 256)
            v = head._predict(x, t, z, cond, dt=torch.full((1,), 0.25),
                              speaker_embedding=spk, cfg_strength=2.0)
            for tag, ten in [("x", x), ("t", t), ("z", z), ("cond", cond),
                             ("spk", spk), ("out", v)]:
                save(f"dit_p{i:02d}_{tag}", ten)
        # B. full sample()
        for i in range(2):
            torch.manual_seed(300 + i)
            z = torch.randn(1, 1024)
            cond = torch.randn(1, 2, 64)
            spk = torch.randn(1, 256)
            out = head.sample(z, cfg=0.0, num_sampling_steps=4, cond=cond,
                              speaker_embedding=spk, sway_sampling_coefficient=0.0,
                              distilled_cfg_strength=2.0)
            save(f"dit_s{i:02d}_z", z)
            save(f"dit_s{i:02d}_cond", cond)
            save(f"dit_s{i:02d}_spk", spk)
            save(f"dit_s{i:02d}_out", out)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
