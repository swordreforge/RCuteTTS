"""Dump M2 full-decode vectors: random latent -> AudioVAE.decode reference.

Run: source ../.venv/bin/activate && python dump_m2_vectors.py
"""
from pathlib import Path

import numpy as np
import torch

from cutetts.audio_codec.model.audio_vae import AudioVAE
from safetensors.torch import load_file

OUT = Path(__file__).resolve().parent.parent / "testdata"
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parent.parent.parent

vae = AudioVAE(
    sample_rate=24000, frame_rate=12.5, vae_dim=64, encoder_dim=128,
    encoder_rates=[3, 5, 8, 16], decoder_dim=1536, decoder_rates=[16, 8, 5, 3],
    depthwise=True, posterior_type="sigma", fix_std=0.15, std_dist_type="gaussian",
)
sd = load_file(ROOT / "model/CuteTTS/weights/audio_vae/model.safetensors")
vae.load_state_dict(sd, strict=True)
vae.eval()

frames_list = [2, 3, 4, 5, 6, 8, 10, 12, 16, 24]
with torch.no_grad():
    for i, nf in enumerate(frames_list):
        torch.manual_seed(100 + i)
        lat = torch.randn(1, 64, nf)
        wav = vae.decode(lat)
        np.save(OUT / f"m2_lat_{i:02d}.npy", np.ascontiguousarray(lat[0].numpy(), np.float32))
        np.save(OUT / f"m2_ref_{i:02d}.npy", np.ascontiguousarray(wav[0].numpy(), np.float32))
        print(f"m2_{i:02d}: frames={nf} samples={wav.shape[-1]}")
print("done ->", OUT)
