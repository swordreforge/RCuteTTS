# CuteTTS Rust 编解码计划书

## 1. 背景

- CuteTTS `tts` 228.6M / distill 231.8M，`+ audio_vae` 127.9M `+ speaker` ~25M，总 ~380M，约为 QORA-TTS-12Hz-1.7B 的 1/4。
- 同机 CPU 实测：QORA RTF 5.0（0.20x），CuteTTS-distill RTF 8.9（0.11x）。参数小但 Python torch + diffusion 反而更慢。
- 论文 `2608.08638v2.pdf`：连续因果 VAE（12.5Hz / 64dim / 压缩 1920x）+ Patch AR（LocEnc 2层 + Qwen3截断7层 + DiT 4层）+ Guidance-Step蒸馏（20 NFE → 1/2/4步）。
- 纠正：`src/cutetts/audio_codec/model/audio_vae.py` 只是复用 DAC 的 CausalConv+Snake 结构，无 RVQ 无码本，输入是 DiT 输出的连续 latent，不是离散 token。

## 2. 目标

- 范围：先只做 `audio_vae.decode` 的 Rust 后端（offline 整句），不碰 LM / DiT / 流式。
- 成功标准：与 `AudioVAE.decode` bit-exact（tol 1e-4）→ PESQ/SIM 持平 → CPU decode 单独 RTF < 0.5 → 端到端 RTF 从 8.9 降到 ~5 以内。
- 非目标：本期不做 DiT 采样循环 Rust 化，不做 VAE encoder / ECAPA / tokenizer，不做量化（decoder 保持 F32，沿用 QORA 结论）。

## 3. 复用 vs 重写（对照 QORA `src/`）

复用（照搬）：
- `conv.rs`：CausalConv1d / ConvTranspose1d / depthwise，`[C,T]` 布局，按 out-channel 并行，500k 阈值。
- `simd.rs`：AVX2 causal-conv 内核（mul+add 保 bit-exact）。
- `gpool.rs`：常驻线程池。
- `decoder.rs`：SnakeBeta / GELU / LayerNorm / RMSNorm / `f32_gemv_bias` / 因果右裁剪 / clamp。
- `save.rs` + `wav.rs` + `chunk.rs`：QTTS 二进制框架、WAV 读写、crossfade。

重写：
1. 输入前端：删 VQ 16码本 + 双 output_proj，换 `latent[64,T] -> hidden` 的 in_proj（以 `model/CuteTTS/weights/audio_vae/config.json` + safetensors 为准）。
2. 上采样拓扑：按 `decoder_dim=1536 rates=[16,8,5,3] depthwise=true` 重写各级 `(in,out,k,stride)`，不能用 QORA 的 `1024→…→96` 写死参数。
3. 流式状态机二期再做：移植 `src/cutetts/modeling/audio_adapter.py:105-232` 的 conv 状态缓存（本期 offline 可跳过）。

## 4. 里程碑

- M0 脚手架（本周）：本仓 `rust-codec/`，`cargo test` 通过，`PLAN.md` 定稿，导出 `model/CuteTTS/weights/audio_vae` 权重清单（shape/dtype）。✅ done：decoder 122 tensors / 45.8M（见 `weights_manifest.json`），对齐向量 `testdata/latent_1x64x8.npy`
- M1 算子对齐：conv / transpose / Snake / norm 单算子 vs torch 差分测试，bit-exact。✅ done：`cargo test` 10/10（dilation 3/9、depthwise、stride 16/3、residual unit，门限 1e-4，见 `tests/m1_ops.rs` + `scripts/dump_m1_fixtures.py`）
- M2 整句 decode 对齐：Python dump `latent[64,T]` → Rust decode → `max_abs_err < 1e-4`，10 条样本全过。✅ done：10/10 全过，worst err 4.2e-7（门限放宽到 1e-3，实际近 bit 级），见 `tests/m2_decode.rs` + `scripts/dump_m2_vectors.py`；naive 单线程基线：0.64s 音频 4.1s（RTF 6.4），M3 优化空间明确
- M3 性能：AVX2 + 线程池，单句 decode RTF < 0.5，与 Python 版同句对比。✅ done：naive 单线程 RTF 6.4 → 并行 + f32 + 边界剥离 + 1x1 SAXPY 后 RTF 0.31（1.92s 音频 0.6s，22线程，`CUTETTS_THREADS` 可调），对齐误差仍 5.6e-7；`target-cpu=native` 无额外收益，未采用
- M4 接入：Python `AudioStreamingVAEDecoder` 后端可切换 `rust|torch`，`inference` 端到端回归（tts + voice_clone各3条）。✅ done（范围：整句 offline `decode()` 经 `CUTETTS_VAE_BACKEND=rust` 切换，见 `src/cutetts/modeling/rust_vae.py` + `src/ffi.rs`；voice_clone 流式仍走 torch）：单元 err 3.9e-7，e2e 3条 tts err ~2e-6，voice_clone 冒烟过。
- M6 GEMM 内核（本轮）：读 QORA `simd.rs`（AVX2 mul+add 保 bit-exact、FMA 另门控）+ `gemv.rs` 调度做对照。发现三点：①本机无 AVX512（155H），QORA 的 512 路径跑不到；②我们之前靠 autovec，默认只编出 SSE（4-wide、无 FMA），MKL 是 AVX2+FMA（8-wide融合），理论差 4x；③`env::var` 写在内层 dispatch 里，整句约 4500 万次调用，全烧在 getenv（m2 全量从 2.6s 涨到 68s，抓包确认）。修法：`src/simd.rs` 手写 SAXPY（exact mul+add 与 scalar bit 一致，FMA 单舍入，`CUTETTS_FMA=0` 可回 exact）+ `resolve_saxpy()` 一次提升。结果：24帧 0.254→0.224s（RTF 0.116，torch 0.209/0.109，差 7%）；120帧 1.153s vs torch 1.202s，反超 4%。FMA 版 err 3.32e-7 反比 exact（3.73e-7）更接近 torch——佐证 MKL 内部也是 FMA。剩 7% 在循环顺序：C tile 每 K 迭代 load+store 一次，寄存器分块（GotoBLAS式 MR×NR，C 常驻 ymm）可补，下一刀。
- M7 寄存器分块（本轮）：8x8 微内核（`simd.rs:micro8_body!`，QORA 宏风格），8 个 ymm acc 横跨整个 K 循环，C 流量 O(K)→O(1)；A 广播、B 8-wide load 各 stream 一次；列余数（<8）走 bias-fill+SAXPY；`sgemm_bias_with_kind` 供测试显式选核。exact 与 scalar 按位一致（op 序相同），FMA 仅舍入差。结果：24帧 0.224→0.162s（RTF 0.085，torch 0.209/0.109，领先 29%）；120帧 0.945s vs torch 1.202s（领先 27%）；M2 10/10 worst 3.46e-7。从 M5 的落后 22% 到领先 ~28%。
- M8 bf16 核（本轮，LM 门票）：`simd.rs:bf16_to_f32_avx2`（load128→cvtepu16→slli16→store256，8-wide），bit 精确（bf16 即 f32 高 16 位，`<<16` 恒等），全 patterns（含 NaN payload）按位一致；`half` cross-check 发现其规范化 NaN（与 torch/CUDA shift 语义不同，权重全有限故无碍，测试限非 NaN）；127M 量级 39ms / 19.6 GB/s（memcpy-bound），加载开销忽略。
- M9 DiT 开工（本轮）：`diffusion_head.py` 全读 + distill 调用形状锁定（`sample(z=[1,1024], cfg=0, steps=4, cond=[1,2,64] 上步 patch, spk=[1,256], w=2.0)`，`mu_proj` 为 Identity 无权重）；fixtures（3×`_predict` + 2×`sample`，`scripts/dump_dit_vectors.py`）+ `dit_manifest.json`；`src/dit.rs` loader（61 tensors / 73.6M fp32，无 weight_norm，直接 pack，`tests/dit_load.rs` 过）。结构：4×[GQA 16Q/2KV RoPE θ=1e4 双向 seq=5 + SwiGLU 4096 + AdaLN-Zero 6144×256] + time/delta/step/cfg 四组 MLP。下一步：`_predict` forward（seq=5 attention 可 scalar，linears 走 sgemm T=5→pad8），time/cfg/step 条件按 sample 预计算（torch 每步重算，白送的加速）。

## 5. 风险

- `weight_norm` 推理需 fuse g/v，`Snake` 逐点三角函数精度漂 → 用 AVX2 mul+add 内核锁 bit。
- 混合精度：LM bf16 / head fp32 / VAE fp32，Rust 侧统一 F32，scale/bias 逆归一别漏（`model.py:95`）。
- `decoder_rates` 与 safetensors 实际 shape 不一致时以权重为准，先打印清单再写 forward。
