"""Record a real tts generate() run (BASE, cpu, seed 42) for the Rust e2e test.

Base dual-branch LM CFG: per AR step the DiT runs 10 sway-Euler steps over
a batched [2,...] (cond row 0 / uncond row 1) with v = vc + 2.0*(vc - vu).
Hooks (monkeypatch, no production changes):
  - AudioDiTHead._predict: per Euler-step x/t/z/cond (batched [2,...])
  - diffusion_head._euler_sway: per AR step final pred_latent
  - CuteTTSModel.forward_lm: inputs_embeds + position_ids + last_hidden
  - generation._stop_after_current_patch: stop flags (cond branch)
Saves: cond/uncond prefixes, per-step x0/zc/zu/cond/uncond/pred, stops,
cond-branch lm in/out, final waveform. PREFIX=e2eb_.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent.parent / "testdata"
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cutetts import CuteTTS  # noqa: E402
from cutetts.inference import generation as gen_mod  # noqa: E402
from cutetts.modeling import diffusion_head as head_mod  # noqa: E402
from cutetts.modeling.diffusion_head import AudioDiTHead  # noqa: E402
from cutetts.modeling.model import CuteTTSModel  # noqa: E402
from cutetts.modeling.sampling import set_sampler_compile_mode  # noqa: E402

set_sampler_compile_mode("eager")

TEXT = os.environ.get("E2E_TEXT", "Hello world, this is a test.")
PREFIX = "e2eb_"
rec = {"pred": [], "lm_in": [], "lm_pos": [], "lm_out": [], "stop": []}

orig_sway = head_mod._euler_sway
def sway_wrapper(*args, **kw):
    out = orig_sway(*args, **kw)
    rec["pred"].append(out.detach().clone())
    return out
head_mod._euler_sway = sway_wrapper

orig_predict = AudioDiTHead._predict
def predict_wrapper(self, x, t, z, cond, dt=None, speaker_embedding=None,
                    cfg_strength=None, validate_conditions=True,
                    fixed_condition_cache=None, fixed_condition_index=None):
    rec.setdefault("px", []).append(x.detach().clone())
    rec.setdefault("pz", []).append(z.detach().clone())
    rec.setdefault("pcond", []).append(cond.detach().clone())
    assert speaker_embedding is None, "tts fixture must have no speaker"
    assert cfg_strength is None, "base has no distilled cfg strength"
    out = orig_predict(self, x, t, z, cond, dt, speaker_embedding, cfg_strength,
                       validate_conditions, fixed_condition_cache, fixed_condition_index)
    return out
AudioDiTHead._predict = predict_wrapper

orig_flm = CuteTTSModel.forward_lm
def flm_wrapper(self, **kw):
    rec["lm_in"].append(kw["inputs_embeds"].detach().clone())
    rec["lm_pos"].append(kw["position_ids"].detach().clone())
    out = orig_flm(self, **kw)
    rec["lm_out"].append(out.last_hidden_state.detach().clone())
    return out
CuteTTSModel.forward_lm = flm_wrapper

orig_stop = gen_mod._stop_after_current_patch
def stop_wrapper(lm, hidden):
    r = orig_stop(lm, hidden)
    rec["stop"].append(bool(r))
    return r
gen_mod._stop_after_current_patch = stop_wrapper


def save(name, t):
    t = np.ascontiguousarray(t.detach().cpu().float().numpy(), dtype=np.float32)
    np.save(OUT / f"{name}.npy", t)
    print(f"{name}: shape={list(t.shape)} maxabs={np.abs(t).max():.4f}")


def row(t, r):
    return t[r:r + 1].reshape(-1).contiguous()


def main() -> None:
    torch.manual_seed(42)
    model = CuteTTS.from_pretrained(ROOT / "model/CuteTTS", device="cpu")
    if os.environ.get("FP32_CAST") == "1":
        m = model.runtime.model
        m.qwen_backbone.float()
        m.locenc.float()
        m.locenc_to_lm_proj.float()
        m.lm_speaker_linear.float()
        m.stop_predictor.float()
        print("FP32_CAST on")
    set_sampler_compile_mode("eager")
    rt = model.runtime
    with torch.no_grad():
        res = model.generate(TEXT, mode="tts", seed=42, show_progress=False)
    n_steps = len(rec["pred"])
    print(f"AR steps: {n_steps}, _predict calls: {len(rec['px'])}, stops: {rec['stop']}")
    assert len(rec["px"]) == 10 * n_steps, len(rec["px"])
    assert len(rec["stop"]) == n_steps
    # lm calls: 2 prefills + 2 decodes per non-final step
    assert len(rec["lm_in"]) == 2 + 2 * (n_steps - 1), len(rec["lm_in"])
    save(f"{PREFIX}prefix", rec["lm_in"][0])
    save(f"{PREFIX}uprefix", rec["lm_in"][1])
    save(f"{PREFIX}prefix_pos", rec["lm_pos"][0].float())
    for i in range(n_steps):
        base = 10 * i
        save(f"{PREFIX}{i:02d}_x0", row(rec["px"][base], 0))
        save(f"{PREFIX}{i:02d}_zc", row(rec["pz"][base], 0))
        save(f"{PREFIX}{i:02d}_zu", row(rec["pz"][base], 1))
        save(f"{PREFIX}{i:02d}_cond", row(rec["pcond"][base], 0))
        save(f"{PREFIX}{i:02d}_uncond", row(rec["pcond"][base], 1))
        save(f"{PREFIX}{i:02d}_pred", rec["pred"][i])
    for i in range(n_steps - 1):
        save(f"{PREFIX}{i:02d}_lmin", rec["lm_in"][2 + 2 * i])
        save(f"{PREFIX}{i:02d}_lmout", rec["lm_out"][2 + 2 * i])
    print("scale:", float(rt.model.speech_scaling_factor), "bias:", float(rt.model.speech_bias_factor))
    with open(OUT.parent / "e2eb_meta.json", "w") as f:
        import json
        json.dump({
            "text": TEXT, "seed": 42, "steps": n_steps,
            "stops": rec["stop"],
            "scale": float(rt.model.speech_scaling_factor),
            "bias": float(rt.model.speech_bias_factor),
            "sample_rate": res.sample_rate,
            "prefix_len": rec["lm_in"][0].shape[1],
        }, f, indent=1)
    save(f"{PREFIX}wav", res.waveform)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
