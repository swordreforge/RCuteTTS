"""Clean Python decode benchmark (no Rust, no FFI).

Usage:
    source .venv/bin/activate
    python rust-codec/scripts/bench_torch_decode.py <latent.npy> [--ref ref.npy] [--repeat N] [--weights ...]

Mirrors rust-codec/src/bin/decode_bench.rs: loads AudioVAE once, warmup once,
then times N whole-sentence decodes. Prints the same summary lines.
"""
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cutetts.audio_codec.model.audio_vae import AudioVAE  # noqa: E402
from safetensors.torch import load_file  # noqa: E402


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(
            "usage: bench_torch_decode.py <latent.npy> [--ref ref.npy] [--repeat N] [--weights ...]",
            file=sys.stderr,
        )
        raise SystemExit(2)
    latent_path = Path(args[0])
    ref_path: Path | None = None
    repeat = 3
    weights = ROOT / "model/CuteTTS/weights/audio_vae/model.safetensors"
    i = 1
    while i < len(args):
        if args[i] == "--ref":
            ref_path = Path(args[i + 1])
            i += 2
        elif args[i] == "--repeat":
            repeat = int(args[i + 1])
            i += 2
        elif args[i] == "--weights":
            weights = Path(args[i + 1])
            i += 2
        else:
            print(f"unknown arg {args[i]}", file=sys.stderr)
            raise SystemExit(2)

    t0 = time.time()
    vae = AudioVAE(
        sample_rate=24000, frame_rate=12.5, vae_dim=64, encoder_dim=128,
        encoder_rates=[3, 5, 8, 16], decoder_dim=1536, decoder_rates=[16, 8, 5, 3],
        depthwise=True, posterior_type="sigma", fix_std=0.15, std_dist_type="gaussian",
    )
    vae.load_state_dict(load_file(weights), strict=True)
    vae.eval()
    load_s = time.time() - t0

    lat = np.load(latent_path).astype(np.float32)
    assert lat.shape[0] == 64, f"latent must be [64, frames], got {lat.shape}"
    frames = lat.shape[1]
    audio_s = frames * 1920 / 24000.0
    tensor = torch.from_numpy(np.ascontiguousarray(lat)).unsqueeze(0)

    threads = torch.get_num_threads()
    with torch.no_grad():
        wav = vae.decode(tensor).float()
        assert wav.shape[-1] == frames * 1920
        times = []
        for n in range(repeat):
            t1 = time.time()
            wav = vae.decode(tensor).float()
            dt = time.time() - t1
            times.append(dt)
            print(f"repeat[{n}]: {dt:.3f}s rtf={dt / audio_s:.3f}", flush=True)
    median = statistics.median(times)
    print(
        f"summary: frames={frames} audio={audio_s:.2f}s threads={threads} "
        f"load={load_s:.2f}s median={median:.3f}s rtf={median / audio_s:.3f}"
    )
    if ref_path is not None:
        ref = np.load(ref_path).astype(np.float32).reshape(-1)
        got = wav[0, 0].numpy().reshape(-1)
        assert ref.shape == got.shape, (ref.shape, got.shape)
        print(f"max_abs_err={np.abs(ref - got).max():.2e}")


if __name__ == "__main__":
    main()
