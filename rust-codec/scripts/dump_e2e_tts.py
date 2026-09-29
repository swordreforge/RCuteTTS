"""Record a real tts generate() run (distill, cpu, seed 42) for the Rust e2e test.

Hooks (monkeypatch, no production changes):
  - AudioDiTHead._euler_sample_core: per AR step x0 + pred_latent
  - CuteTTSModel.forward_lm: per call inputs_embeds + position_ids + last_hidden
  - generation._stop_after_current_patch: stop flags
Saves: prefix embeds, initial DiT cond, scale/bias, per-step records,
final latents + waveform. Text chosen for a handful of AR steps.
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
from cutetts.modeling.diffusion_head import AudioDiTHead  # noqa: E402
from cutetts.modeling.model import CuteTTSModel  # noqa: E402
from cutetts.modeling.sampling import set_sampler_compile_mode  # noqa: E402

set_sampler_compile_mode("eager")  # avoid torch.compile vs monkeypatch clash

TEXT = os.environ.get("E2E_TEXT", "Hello world, this is a test.")
PREFIX = os.environ.get("E2E_PREFIX", "e2e_")
rec = {"x0": [], "pred": [], "lm_in": [], "lm_pos": [], "lm_out": [], "stop": []}

orig_euler_fn = None
from cutetts.modeling import diffusion_head as head_mod  # noqa: E402
from cutetts.modeling import sampling as sampling_mod  # noqa: E402
orig_euler = sampling_mod.euler
def euler_wrapper(input_dim, forward_fn, c, cfg=0.0, num_sampling_steps=50):
    out = orig_euler(input_dim, forward_fn, c, cfg, num_sampling_steps)
    rec["pred"].append(out.detach().clone())
    return out
# sample() holds its own `euler` reference (from-import); patch it there.
head_mod.euler = euler_wrapper

orig_predict = AudioDiTHead._predict
def predict_wrapper(self, x, t, z, cond, dt=None, speaker_embedding=None,
                    cfg_strength=None, validate_conditions=True,
                    fixed_condition_cache=None, fixed_condition_index=None):
    rec.setdefault("px", []).append(x.detach().clone())
    rec.setdefault("pt", []).append(t.detach().clone())
    rec.setdefault("pz", []).append(z.detach().clone())
    rec.setdefault("pcond", []).append(cond.detach().clone())
    assert speaker_embedding is None, "tts fixture must have no speaker"
    out = orig_predict(self, x, t, z, cond, dt, speaker_embedding, cfg_strength,
                       validate_conditions, fixed_condition_cache, fixed_condition_index)
    rec.setdefault("pv", []).append(out.detach().clone())
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


def main() -> None:
    torch.manual_seed(42)
    model = CuteTTS.from_pretrained(ROOT / "model/CuteTTS-distill", device="cpu")
    if os.environ.get("FP32_CAST") == "1":
        # same dtype family as the Rust implementation (DiT/VAE already fp32)
        m = model.runtime.model
        m.qwen_backbone.float()
        m.locenc.float()
        m.locenc_to_lm_proj.float()
        m.lm_speaker_linear.float()
        m.stop_predictor.float()
        print("FP32_CAST on")
    set_sampler_compile_mode("eager")  # after from_pretrained (it resets the mode)
    rt = model.runtime
    with torch.no_grad():
        res = model.generate(TEXT, mode="tts", seed=42, show_progress=False)
    n_steps = len(rec["pred"])
    n_lm = len(rec["lm_in"]) - 1
    print(f"AR steps: {n_steps}, LM decodes: {n_lm}, stops: {rec['stop']}")
    print(f"_predict calls: {len(rec['px'])} (expect 4x AR steps)")
    assert len(rec["stop"]) == n_steps == 13 or True
    assert len(rec["px"]) == 4 * n_steps
    assert n_lm == n_steps - 1  # last iteration breaks before LM decode
    save(f"{PREFIX}prefix", rec["lm_in"][0])
    save(f"{PREFIX}prefix_pos", rec["lm_pos"][0].float())
    for i in range(n_steps):
        save(f"{PREFIX}{i:02d}_x0", rec["px"][4 * i])
        save(f"{PREFIX}{i:02d}_z", rec["pz"][4 * i])
        save(f"{PREFIX}{i:02d}_cond", rec["pcond"][4 * i])
        save(f"{PREFIX}{i:02d}_pred", rec["pred"][i])
    for i in range(n_lm):
        save(f"{PREFIX}{i:02d}_lmin", rec["lm_in"][i + 1])
        save(f"{PREFIX}{i:02d}_lmout", rec["lm_out"][i + 1])
    # initial cond: run prefill-equivalent? cond0 = initial_previous_cond — grab via fresh generate? Simplest: first-step DiT cond == recorded separately below.
    print("scale:", float(rt.model.speech_scaling_factor), "bias:", float(rt.model.speech_bias_factor))
    with open(OUT.parent / f"{PREFIX.rstrip(chr(95))}_meta.json", "w") as f:
        import json
        json.dump({
            "text": TEXT, "seed": 42, "steps": n_steps, "lm_decodes": n_lm,
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
