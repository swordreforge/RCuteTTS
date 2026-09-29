# Copyright 2026 OPPO and Fudan University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rust whole-sentence VAE decode backend (ctypes -> rust-codec cdylib).

Select with ``CUTETTS_VAE_BACKEND=rust``. Library lookup order:
``CUTETTS_CODEC_LIB`` env, then ``rust-codec/target/release/libcutetts_codec.so``
relative to the CuteTTS repo root.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch

HOP_LENGTH = 1920
VAE_DIM = 64


def _default_lib() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parents[3], here.parents[2]):
        candidate = parent / "rust-codec" / "target" / "release" / "libcutetts_codec.so"
        if candidate.is_file():
            return candidate
    return parent / "rust-codec" / "target" / "release" / "libcutetts_codec.so"


def _load_cdylib() -> ctypes.CDLL:
    override = os.environ.get("CUTETTS_CODEC_LIB")
    path = Path(override).expanduser() if override else _default_lib()
    if not path.is_file():
        raise FileNotFoundError(
            f"Rust codec library not found at {path} "
            "(build with `cargo build --release` in rust-codec/)."
        )
    lib = ctypes.CDLL(str(path))
    lib.cutetts_vae_load.argtypes = [ctypes.c_char_p]
    lib.cutetts_vae_load.restype = ctypes.c_void_p
    lib.cutetts_vae_free.argtypes = [ctypes.c_void_p]
    lib.cutetts_vae_free.restype = None
    lib.cutetts_vae_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.cutetts_vae_decode.restype = ctypes.c_int
    return lib


class RustVAEDecoder:
    """Whole-sentence decode only; streaming still uses torch."""

    def __init__(self, weights_path: str | Path):
        self._lib = _load_cdylib()
        handle = self._lib.cutetts_vae_load(str(weights_path).encode())
        if not handle:
            raise RuntimeError(f"cutetts_vae_load failed for {weights_path}")
        self._handle = handle

    def __del__(self):
        try:
            if getattr(self, "_handle", None):
                self._lib.cutetts_vae_free(self._handle)
                self._handle = None
        except Exception:
            pass

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.dim() != 3:
            raise ValueError(f"Expected 3D latent, got {tuple(latent.shape)}.")
        if latent.shape[-1] == VAE_DIM:
            latent_cf = latent.transpose(1, 2)
        elif latent.shape[1] == VAE_DIM:
            latent_cf = latent
        else:
            raise ValueError(f"Cannot infer latent layout for {tuple(latent.shape)}.")
        latent_cf = latent_cf.detach().to(device="cpu", dtype=torch.float32).contiguous()
        batch = latent_cf.shape[0]
        outs = []
        for b in range(batch):
            frames = latent_cf.shape[2]
            inp = latent_cf[b].numpy()
            out = torch.empty(frames * HOP_LENGTH, dtype=torch.float32).numpy()
            rc = self._lib.cutetts_vae_decode(
                self._handle,
                inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                frames,
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            if rc != 0:
                raise RuntimeError("cutetts_vae_decode failed.")
            outs.append(torch.from_numpy(out).unsqueeze(0).unsqueeze(0))
        return torch.cat(outs, dim=0)
