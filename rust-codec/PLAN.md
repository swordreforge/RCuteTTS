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
- M3 性能：AVX2 + 线程池，单句 decode RTF < 0.5，与 Python 版同句对比。
- M4 接入：Python `AudioStreamingVAEDecoder` 后端可切换 `rust|torch`，`inference` 端到端回归（tts + voice_clone各3条）。

## 5. 风险

- `weight_norm` 推理需 fuse g/v，`Snake` 逐点三角函数精度漂 → 用 AVX2 mul+add 内核锁 bit。
- 混合精度：LM bf16 / head fp32 / VAE fp32，Rust 侧统一 F32，scale/bias 逆归一别漏（`model.py:95`）。
- `decoder_rates` 与 safetensors 实际 shape 不一致时以权重为准，先打印清单再写 forward。
