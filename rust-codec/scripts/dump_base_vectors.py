"""Dump BASE DiT fixtures: torch -> npy for Rust tests (P1).

Base differs from distill: no step_size/cfg_strength embeddings (zeros),
sway Euler grid (default coeff -0.8, 10 steps), dual-branch CFG combine
v = vc + 2.0*(vc - vu), separate cond/uncond previous patches.

Covers:
  A. sway_timesteps(10, -0.8) grid (11 values)
  B. unit _predict, base path (dt=None, no cfg_strength), 2 seeds
  C. full sample() (steps=10, cfg=2.0, sway=-0.8), 2 seeds, x0 replayable
  D. base tts prefixes: cond (text prompt) + uncond (suffix token only)

Run: source ../.venv/bin/activate && python dump_base_vectors.py
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
from cutetts.modeling.diffusion_head import AudioDiTHead, sway_timesteps  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

WEIGHTS = ROOT / "model/CuteTTS/weights/tts/model.safetensors"
TEXT = "Hello world, this is a test."


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
        cfg_strength_embedding_enabled=False,
        cfg_strength_max=5.0,
        step_size_embedding_enabled=False,
        step_schedule="uniform",
    )
    sd = load_file(WEIGHTS)
    head_sd = {k[len("head."):]: v for k, v in sd.items() if k.startswith("head.")}
    assert len(head_sd) == 55, len(head_sd)
    missing, unexpected = head.load_state_dict(head_sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    print(f"base head: {len(head_sd)} tensors")
    return head.eval()


def save(name, t):
    if isinstance(t, torch.Tensor) and t.dtype == torch.bfloat16:
        t = t.float()
    t = np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32)
    np.save(OUT / f"{name}.npy", t)
    print(f"{name}: shape={list(t.shape)} maxabs={np.abs(t).max():.4f}")


def main() -> None:
    head = build_head()
    with torch.no_grad():
        # A. sway grid
        times = sway_timesteps(10, -0.8)
        save("base_sway", times)
        # B. unit _predict (base path: dt=None, cfg_strength=None)
        for i in range(2):
            torch.manual_seed(500 + i)
            x = torch.randn(1, 2, 64)
            t = torch.full((1,), 0.3)
            z = torch.randn(1, 1024)
            cond = torch.randn(1, 2, 64)
            v = head._predict(x, t, z, cond)
            for tag, ten in [("x", x), ("t", t), ("z", z), ("cond", cond), ("out", v)]:
                save(f"base_p{i:02d}_{tag}", ten)
        # C. explicit sway Euler (mirrors _euler_sway, patch_size==2):
        # x0 = randn, then 10x _predict on the sway grid with CFG combine
        # v = vc + 2.0*(vc - vu). No RNG inside the loop; x0 saved for replay.
        times = sway_timesteps(10, -0.8)
        for i in range(2):
            torch.manual_seed(600 + i)
            zc = torch.randn(1, 1024)
            zu = torch.randn(1, 1024)
            cond = torch.randn(1, 2, 64)
            uncond = torch.randn(1, 2, 64)
            x0 = torch.randn(1, 2, 64)
            x = x0.clone()
            for step in range(10):
                # per-branch [1,...] calls: rows are independent in _predict,
                # exactly equal to sample()'s batched [2,...] call
                t = torch.full((1,), times[step].item())
                vc = head._predict(x, t, zc, cond, validate_conditions=False)
                vu = head._predict(x, t, zu, uncond, validate_conditions=False)
                v = vc + 2.0 * (vc - vu)
                dt = times[step + 1] - times[step]
                x = x + v * dt
            save(f"base_s{i:02d}_zc", zc)
            save(f"base_s{i:02d}_zu", zu)
            save(f"base_s{i:02d}_cond", cond)
            save(f"base_s{i:02d}_uncond", uncond)
            save(f"base_s{i:02d}_x0", x0)
            save(f"base_s{i:02d}_out", x)
    # D. real tts prefixes (cond + 1-token uncond)
    model = CuteTTS.from_pretrained(ROOT / "model/CuteTTS", device="cpu")
    plan = build_guidance_plan("tts", "lm", 2.0)
    assert plan.uses_lm_cfg and plan.unconditional is not None
    with torch.no_grad():
        cseg = build_prefix_segment(model.runtime.processor, plan.conditional, target_text=TEXT)
        useg = build_prefix_segment(model.runtime.processor, plan.unconditional, target_text=TEXT)
        print(f"cond total={cseg.total_length} uncond total={useg.total_length}")
        cemb, _, _ = model.runtime.model.prepare_input_embeds(cseg, lm_speaker_embedding=None)
        uemb, _, _ = model.runtime.model.prepare_input_embeds(useg, lm_speaker_embedding=None)
    save("base_cond_prefix", cemb[0])
    save("base_uncond_prefix", uemb[0])
    print("done ->", OUT)


if __name__ == "__main__":
    main()
