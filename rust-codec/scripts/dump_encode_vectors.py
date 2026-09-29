"""Dump VAE-encoder fixtures: AudioVAE.encode -> mu (mode, no sampling).

Inference uses posterior mode only (acoustic_feature_no_sampling=True);
fc_logvar is dead at inference. Cases: exact-multiple lengths, a
non-multiple (preprocess right-pad), and the real default_reference
resampled to 24k (production clone input). Saves wave (fp32 mono) + mu.
"""
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent.parent / "testdata"
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cutetts.audio_codec.model.audio_vae import AudioVAE  # noqa: E402
from cutetts.runtime import prepare_reference_audio  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

WAV = ROOT / "assets/default_reference.wav"


def build():
    vae = AudioVAE(
        sample_rate=24000, frame_rate=12.5, vae_dim=64, encoder_dim=128,
        encoder_rates=[3, 5, 8, 16], decoder_dim=1536, decoder_rates=[16, 8, 5, 3],
        depthwise=True, posterior_type="sigma", fix_std=0.15, std_dist_type="gaussian",
    )
    sd = load_file(ROOT / "model/CuteTTS/weights/audio_vae/model.safetensors")
    vae.load_state_dict(sd, strict=True)
    return vae.eval()


def save(name, t):
    t = np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32)
    np.save(OUT / f"{name}.npy", t)
    print(f"{name}: shape={list(t.shape)} maxabs={np.abs(t).max():.4f}")


def main() -> None:
    vae = build()
    manifest = {}
    sd = load_file(ROOT / "model/CuteTTS/weights/audio_vae/model.safetensors")
    for k, v in sorted(sd.items()):
        if k.startswith("encoder.") and not k.startswith("encoder.fc_logvar"):
            manifest[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
    import json
    with open(OUT.parent / "vae_enc_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"encoder tensors (no logvar): {len(manifest)}")

    torch.manual_seed(11)
    waves = {
        "00": torch.randn(1, 1, 3840),          # exact 2 frames
        "01": torch.randn(1, 1, 9700),          # non-multiple -> pad
        "02": torch.randn(1, 1, 24000),         # 1s -> 13 frames (pad)
    }
    with torch.no_grad():
        for tag, wv in waves.items():
            pre = vae.preprocess(wv)
            mu = vae.encode(pre).mode()[0]  # [T,64]
            save(f"enc_{tag}_wav", wv[0, 0])
            save(f"enc_{tag}_mu", mu)
    # production reference input
    with torch.no_grad():
        ref_wave, _ = prepare_reference_audio(WAV, 24000, 16000)
        mu = vae.encode(vae.preprocess(ref_wave.unsqueeze(0))).mode()[0]
    save("enc_ref_wav", ref_wave)
    save("enc_ref_mu", mu)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
