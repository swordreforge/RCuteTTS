"""Record ECAPA fixtures + a real voice_clone generate() run (distill, cpu).

ECAPA part (fp32 both sides -> tight gates):
  - speaker_wave: prepare_reference_audio(default_reference.wav)[1] (16k)
  - log-mel via feature_extract, embedding via speaker_encoder
  - second vector: first half of the wave (different length)
Clone ring part: same hook set as dump_e2e_tts (euler/_predict/flm/stop),
  mode=voice_clone, same text/seed. Saves clone_* fixtures + clone_meta.json.
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
from cutetts.inference import generation as gen_mod  # noqa: E402
from cutetts.modeling import diffusion_head as head_mod  # noqa: E402
from cutetts.modeling.diffusion_head import AudioDiTHead  # noqa: E402
from cutetts.modeling.model import CuteTTSModel  # noqa: E402
from cutetts.modeling.sampling import set_sampler_compile_mode  # noqa: E402
from cutetts.runtime import prepare_reference_audio  # noqa: E402

TEXT = "Hello world, this is a test."
REF = ROOT / "assets/default_reference.wav"
rec = {"pred": [], "lm_in": [], "lm_pos": [], "lm_out": [], "stop": []}


def save(name, t):
    t = np.ascontiguousarray(t.detach().cpu().float().numpy(), dtype=np.float32)
    np.save(OUT / f"{name}.npy", t)
    print(f"{name}: shape={list(t.shape)} maxabs={np.abs(t).max():.4f}")


def main() -> None:
    torch.manual_seed(7)
    model = CuteTTS.from_pretrained(ROOT / "model/CuteTTS-distill", device="cpu")
    set_sampler_compile_mode("eager")
    rt = model.runtime

    # --- ECAPA fixtures ---
    ref_wave, spk_wave = prepare_reference_audio(
        REF, rt.sample_rate, int(rt.speaker_encoder.sample_rate))
    print(f"speaker wave: {tuple(spk_wave.shape)} @ {int(rt.speaker_encoder.sample_rate)}Hz")
    save("spk_wave", spk_wave)
    with torch.no_grad():
        mel = rt.speaker_encoder.feature_extract(spk_wave)
        out = rt.speaker_encoder(spk_wave, int(rt.speaker_encoder.sample_rate))
    save("spk_mel", mel)
    save("spk_emb", out["embedding"])
    half = spk_wave[..., : spk_wave.shape[-1] // 2].contiguous()
    with torch.no_grad():
        out_h = rt.speaker_encoder(half, int(rt.speaker_encoder.sample_rate))
    save("spk_wave_half", half)
    save("spk_emb_half", out_h["embedding"])

    # --- clone ring hooks ---
    orig_euler = head_mod.euler
    def euler_wrapper(input_dim, forward_fn, c, cfg=0.0, num_sampling_steps=50):
        out = orig_euler(input_dim, forward_fn, c, cfg, num_sampling_steps)
        rec["pred"].append(out.detach().clone())
        return out
    head_mod.euler = euler_wrapper

    orig_predict = AudioDiTHead._predict
    def predict_wrapper(self, x, t, z, cond, dt=None, speaker_embedding=None,
                        cfg_strength=None, validate_conditions=True,
                        fixed_condition_cache=None, fixed_condition_index=None):
        rec.setdefault("px", []).append(x.detach().clone())
        rec.setdefault("pt", []).append(t.detach().clone())
        rec.setdefault("pz", []).append(z.detach().clone())
        rec.setdefault("pcond", []).append(cond.detach().clone())
        rec.setdefault("pspk", []).append(speaker_embedding.detach().clone())
        assert speaker_embedding is not None, "clone fixture must have speaker"
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

    torch.manual_seed(42)
    with torch.no_grad():
        res = model.generate(TEXT, mode="voice_clone", reference_audio=str(REF),
                             seed=42, show_progress=False)
    n_steps = len(rec["pred"])
    n_lm = len(rec["lm_in"]) - 1
    print(f"AR steps: {n_steps}, LM decodes: {n_lm}, stops: {rec['stop']}")
    assert len(rec["px"]) == 4 * n_steps and n_lm == n_steps - 1
    save("clone_prefix", rec["lm_in"][0])
    for i in range(n_steps):
        save(f"clone_{i:02d}_x0", rec["px"][4 * i])
        save(f"clone_{i:02d}_z", rec["pz"][4 * i])
        save(f"clone_{i:02d}_cond", rec["pcond"][4 * i])
        save(f"clone_{i:02d}_spk", rec["pspk"][4 * i])
        save(f"clone_{i:02d}_pred", rec["pred"][i])
    for i in range(n_lm):
        save(f"clone_{i:02d}_lmin", rec["lm_in"][i + 1])
        save(f"clone_{i:02d}_lmout", rec["lm_out"][i + 1])
    with open(OUT.parent / "clone_meta.json", "w") as f:
        import json
        json.dump({
            "text": TEXT, "seed": 42, "steps": n_steps, "lm_decodes": n_lm,
            "stops": rec["stop"],
            "scale": float(rt.model.speech_scaling_factor),
            "bias": float(rt.model.speech_bias_factor),
            "sample_rate": res.sample_rate,
            "prefix_len": rec["lm_in"][0].shape[1],
            "speaker_rate": int(rt.speaker_encoder.sample_rate),
        }, f, indent=1)
    save("clone_wav", res.waveform)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
