"""Dump M1 differential fixtures: torch (audio_vae.py) -> npy for Rust tests.

Run: source ../.venv/bin/activate && python dump_m1_fixtures.py
Output: ../testdata/m1_*.npy (float32, channel-first [C, T] / [O, I, K]).
"""
import math
from pathlib import Path

import numpy as np
import torch

from cutetts.audio_codec.model.audio_vae import (
    CausalResidualUnit,
    Snake1d,
    WNCausalConv1d,
    WNCausalTransposeConv1d,
)

OUT = Path(__file__).resolve().parent.parent / "testdata"
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(7)


def save(name, t):
    t = np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32)
    np.save(OUT / f"{name}.npy", t)
    print(f"{name}: shape={list(t.shape)} maxabs={np.abs(t).max():.4f}")


def dump_conv(name, mod, c_in, t_in):
    mod.eval()
    x = torch.randn(1, c_in, t_in)
    with torch.no_grad():
        y = mod(x)
    save(f"m1_{name}_in", x[0])
    save(f"m1_{name}_w", mod.weight.detach())
    save(f"m1_{name}_b", mod.bias.detach())
    save(f"m1_{name}_out", y[0])


# A: dilated grouped?=1 conv k=7 d=3 (residual-unit style, dim=6)
dump_conv("conv_d3", WNCausalConv1d(6, 6, kernel_size=7, dilation=3, padding=9), 6, 11)
# B: depthwise conv k=7 d=9 groups=8 (decoder residual style)
dump_conv(
    "conv_dw9",
    WNCausalConv1d(8, 8, kernel_size=7, dilation=9, padding=27, groups=8),
    8,
    13,
)
# C: transpose even stride (decoder stage0 style, scaled down): k=32 s=16
dump_conv(
    "trans_s16",
    WNCausalTransposeConv1d(8, 4, kernel_size=32, stride=16, padding=8, output_padding=0),
    8,
    5,
)
# D: transpose odd stride: k=6 s=3 pad=2 out_pad=1
dump_conv(
    "trans_s3",
    WNCausalTransposeConv1d(6, 4, kernel_size=6, stride=3, padding=2, output_padding=1),
    6,
    7,
)
# E: full residual unit dim=8 d=3 (dump sub-conv fused weights separately)
resunit = CausalResidualUnit(dim=8, dilation=3).eval()
xr = torch.randn(1, 8, 11)
with torch.no_grad():
    yr = resunit(xr)
save("m1_resunit_d3_in", xr[0])
save("m1_resunit_d3_w0", resunit.block[1].weight.detach())
save("m1_resunit_d3_b0", resunit.block[1].bias.detach())
save("m1_resunit_d3_w1", resunit.block[3].weight.detach())
save("m1_resunit_d3_b1", resunit.block[3].bias.detach())
save("m1_resunit_d3_out", yr[0])
# F: snake
snake = Snake1d(6).eval()
xs = torch.randn(1, 6, 9)
with torch.no_grad():
    ys = snake(xs)
save("m1_snake_in", xs[0])
save("m1_snake_alpha", snake.alpha.detach()[0, :, 0])
save("m1_snake_out", ys[0])

print("done ->", OUT)
