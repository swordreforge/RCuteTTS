"""Dump resample fixtures: torch sinc_interp_hann (defaults) pairs.

Covers production conversions (prepare_reference_audio): source -> 24000
(reference) and source -> 16000 (speaker). Inputs synthesized by torch
from the 24k default reference (deterministic); the TEST compares our
sinc implementation against torch's output on the same inputs.
Target length rule (ceil) and 24k->24k passthrough included.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as F

OUT = Path(__file__).resolve().parent.parent / "testdata"
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

SRC = ROOT / "assets/default_reference.wav"

CASES = [
    ("44100_24000", 44100, 24000),
    ("48000_24000", 48000, 24000),
    ("48000_16000", 48000, 16000),
    ("44100_16000", 44100, 16000),
    ("24000_16000", 24000, 16000),
    ("16000_24000", 16000, 24000),
    ("22050_24000", 22050, 24000),
    ("24000_24000", 24000, 24000),  # passthrough
]


def main() -> None:
    import soundfile as sf
    wav, sr = sf.read(SRC, dtype="float32", always_2d=True)
    assert sr == 24000
    base = torch.from_numpy(wav.T.copy()).mean(dim=0, keepdim=True)[:, :24000]  # 1s mono
    for tag, orig, new in CASES:
        if orig == 24000:
            src = base
        else:
            with torch.no_grad():
                src = F.resample(base, 24000, orig)
        with torch.no_grad():
            got = F.resample(src, orig, new)
        np.save(OUT / f"rs_{tag}_in.npy", np.ascontiguousarray(src.numpy(), np.float32))
        np.save(OUT / f"rs_{tag}_out.npy", np.ascontiguousarray(got.numpy(), np.float32))
        print(f"rs_{tag}: {orig}->{new} in={list(src.shape)} out={list(got.shape)}")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
