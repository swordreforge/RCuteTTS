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
- M10 DiT forward（本轮）：`src/dit.rs predict/euler_sample`，linears 经 `linear_rows`（转置+pad8 进微内核，大 MLP 线性多线程、小线性串行），attention 5×5 scalar + GQA repeat，RoPE 表预计算，AdaLN modulate/gate，`sample_prologue` 把 adaln/time/cfg/step 一次算完（torch 每步重算）。`tests/dit_predict.rs`：3×predict worst 5.6e-6，2×euler(4步) worst 2.1e-6；22线程 40.4ms vs torch eager 43ms（打平）。单线程 50-70ms（torch 用 16 线程，并无优势）。余量（后压）：scratch 复用（480 allocs/sample）、attention SIMD；先不动，LM 占 99% 才是正事。
- M11 LocEnc（本轮）：生产跑 bf16（`model.py:92`），含 `locenc_to_lm_proj` 共 24 tensors；fixtures bf16/fp32 双跑（`dump_locenc_vectors.py`，3D 输入在 patch=2 下是死路径，只留 4D）；`src/locenc.rs`：loader 经 M8 bf16 核转 fp32（首次实战），forward 复用 DiT 件（`linear_rows` 加 `pub(crate)`，RoPE 改参量化 `rope_tables_for`，新增 plain 无 AdaLN 层），`linear_rows_in` 按 8 行分块（prefill 行数可达数百，batch=2 case 抓到越界）。对齐：3/3 worst 9.06e-6（fp32 紧门限）；bf16 噪声地板 3~8e-2（输出 ~8，约 1%），parity 时再处理。下一步：Qwen3-7L（KV-cache + GQA 16/8 + RoPE θ=1e6 + sampler）。
- M12 Qwen3 深挖（本轮）：新鲜 7 层（非截断，`model.py` 构造时改层数），head_dim **128**，QK-Norm（q/k_norm [128]，QORA talker 同款），GQA 16/8，无 lm_head，**无 token 采样**（stop 靠 Linear，二值分类）——sampler 不用写；mask 全 None，走 DynamicCache；prefill 1 次 + decode 每步 `[1,1,1024]`。fixtures（prefill T=48 + 链式 decode 3 步，fp32/bf16 双跑）+ `qwen_manifest.json`；`src/qwen.rs` loader（79 tensors / 126.9M，0.39s）。bf16 地板 2.8e-1（输出 ~19，1.5%，7 层累积）。下一步：forward（prefill GEMM 路 + decode GEMV 核 + RoPE θ=1e6/d128 + KV-cache）。
- M13 Qwen forward（本轮）：`prefill`（GEMM 路）+ `decode_step`（GEMV `gemv8`112 核，T=1 不 pad）+ KV-cache（f32 存，post-RoPE）；`linear_rows` 改单次 sgemm（旧 8 行分块每块 spawn 22 线程，prefill 全是建线程开销，0.130→0.072s）。对齐：prefill err 1.2e-4，decode ~6e-5，逐层 1e-5~6e-5。速度：prefill T=48 torch 0.04s vs rust 0.072s（MKL 线程多，持久池可再追）；decode torch 13ms vs rust 19.5ms（串行 GEMV，按设计； threading 在小 GEMV 上全是 overhead，不做）。破案记录：①attn_out 按 1024 stride 写进 2048 宽 buffer（T=1 时 i=0 恒对，T≥2 才暴露）；②transformers 4.51 的 hidden_states 是 pre-layer 语义，hs[7] 为 norm 输出非 l6 输出（h7 fixture 已替换为 hook 抓的真值）；③手动调 attention 传 mask=None 等于双向，tok0 参考污染（测试只断言因果干净量）。教训：T=1 测试对 Q/K/stride 类 bug 全盲，必须有 T≥2 + 逐层。
- M14 常驻线程池（本轮）：`src/pool.rs`（workers + queue + counter/condvar，`scope`借入 borrowed 闭包，join 担保 soundness；`notify_all`——单 notify 会饿死并发 waiter，合跑挂 60s+ 抓到）。sgemm 两条路 + decode GEMV（≥1M ops）切池。结果：prefill 0.072→0.061s，decode 19.5→10.8ms（超 torch 13ms 17%），DiT 40.4→30.8ms（超 torch 43ms 28%），全门限按位一致。prefill 距 torch（0.04s）还差 1.5x——剩的是转置 memcpy + MKL blocking，非 spawn，要么持久化布局要么认，留待端到端后再看。
- M15 端到端 Rust 环（本轮，distill tts offline）：DiT 补 spk=None plain 路（对向量 1e-5）；`src/e2e.rs`（stop/scale/bias loader + 环驱动）；`dump_e2e_tts.py` monkeypatch 真实 generate（13 步 DiT + 12 次 LM + 49920 点；eager 下走 `sampling.euler`，`from-import` 绑定要 patch 到 head 命名空间）。关键判定：torch 自加 σ=0.05 前缀噪声即发散 0.615 且变长度——环本性混沌，逐点 parity 无意义。验证策略：teacher-forced（DiT 2.1e-5，wav 2.2e-5，全过）+ 真环 stop 序列 13/13（漂移 0.16→11 如实记录）。教训：VAE 输入是 `[C,T]` 不是 `[T,C]`（0.57 的坑）。
- M16 voice_clone（本轮）：`src/speaker.rs`（221 tensors fp32；STFT reflect/hand-periodic/窗居中/im2col+sgemm；Res2 conv→relu→BN 反常规顺序；ASP 无偏方差；batch=1 按 eval 公式）；ECAPA emb 2.2e-5（fp32 双边，紧门限），0.12s（torch 0.07s，一次性）；clone 环 18 步 teacher-forced（adaln DiT 1.5e-5，wav 5.5e-5）+ 真环 stop（3 处晚期翻转，全在 drift>3 混沌区，门限按漂移挂钩——torch 自噪翻 stop 且长度差 10 倍为证）。破案：hann 漏 cos（-1.7 的窗）；窗摆对信号没跟上；log 门限在 -9dB 处不讲理（0.7% 能量差）；torch hann 是 fp32 逐步舍入（门限 1e-7）。

## 5. 风险

- `weight_norm` 推理需 fuse g/v，`Snake` 逐点三角函数精度漂 → 用 AVX2 mul+add 内核锁 bit。
- 混合精度：LM bf16 / head fp32 / VAE fp32，Rust 侧统一 F32，scale/bias 逆归一别漏（`model.py:95`）。
- `decoder_rates` 与 safetensors 实际 shape 不一致时以权重为准，先打印清单再写 forward。

## 6. 门限审计与测试铁律（M17 起执行）

门限只许和测出来的噪声源挂钩。历史放宽记录：

| 门限 | 变更 | 实测 |  verdict |
|---|---|---|---|
| simd saxpy FMA 1e-6→1e-5 | 单 op cancellation 1.37e-6 | 数学可算 | ✅ 留，注释已写 |
| simd gemv8 FMA 1e-6→5e-6 | K=1024 累积漂移 1.08e-6 | 数学可算 | ✅ 留 |
| hann 1e-9→1e-7 | fp32-vs-f64 7e-9 | 跨库计算顺序 | ✅ 留 |
| speaker mel 1e-3→5e-2 | 只看了 1-2 个点 | **错** | ❌ 已改能量相对门限 1e-4（实测 1.17e-6），旧门限盖住了响部 2.3e-3——回头看全是 floor 线附近，虚惊，但“只看最差点”是错方法 |
| qwen 1e-2（出生即松） | fp32 实际 1e-4 | 门限比实测松 100x | ✅ 已收紧 1e-3 |
| locenc/dit 1e-3（出生即松） | 实际 5~9e-6 | 松 100x+ | ✅ 已收紧 1e-4 |
| m2 1e-3 | 实际 4.2e-7 | 松 2000x | ✅ 已收紧 1e-5 |
| e2e teacher wav 1e-3 | 实际 2.2e-5 | 松 45x | ✅ 已收紧 3e-4 |

铁律：
1. 新测试门限初值 = 首测实测值的 10~100x，不许出生即松 1000x。
2. 放宽门限必须附带 noise-floor 实验证据（oracle-vs-oracle，如 torch-bf16-vs-fp32），写进测试注释。
3. 单点 max 超标时，先看全分布（histogram/分位点/位置），再定是 floor 还是系统偏差——mel 事件教训。
4. 每个 milestone 收紧余量 >100x 的门限（本次已执行）。
5. 混沌环（e2e 真环）只许断言 stop 序列这类鲁棒量，不许断言波形逐点（torch 自噪实验为证）。
6. 行主序的“整段拷贝”一律按行写循环（flat copy 只在行宽相等时成立）——stream `Hist::extend` 与两次测试脚手架同坑三犯。
7. sgemm 语义：bias 按输出行全列折叠，任何“只加一次”的优化先证语义不变。
8. Latent 三形态（raw / scaled / cond）逐处对源码点名：DiT cond=RAW 上步输出，VAE=scaled，LM 回灌=RAW（:1022）；测试必须给回灌加门限——stop 对不代表音频对（M19 血案）。

## 7. 流式（M17）

`src/stream.rs`：显式 history 状态机（零 monkey-patch），conv 走 `valid_conv1d`（history 必须精确 (k-1)*d），转置走 GEMM 分解 + 行切片 `[s..]`。`tests/stream.rs`：2 帧 chunk（AR patch 尺寸）worst 3.5e-7，奇数 chunk 3/5/7 全过，e2e 真实轨迹 1.2e-6，单 chunk vs offline 1.3e-7（近 bit-identical，符合设计）。首包（2 帧）：109→71ms（小-T GEMV 路，T<8 微内核永不点火，改按行向量化）。
破案：①`cv()` 自带因果 pad，传 extended 进去等于 pad 两次——加无 pad 的 `valid_conv1d`；②`Hist::extend` flat-copy 行错位（生产代码真 bug，同类第三次）；③small-T 路 bias 只加 j=0 列，违 sgemm 语义（19 vs 20 差的正好是 bias）。教训见 §6.6/6.7。

## 8. 推理 CLI（M18，本轮）
`src/bin/cutetts.rs`：tokenize → embed → prefill → AR 环 → VAE → wav，零 Python。`src/tok.rs`（sentencepiece crate + 三特殊串先切后 encode，4 文本 id-for-id 精确命中——tricky 证明整串 encode 上下文相关）；`qwen.embed_lookup` 按位 exact；自研 xorshift+Box-Muller RNG（mean -0.0004/var 0.9994，有单测锁死——第一版 uniform 写成 [1,2) 全 NaN 跑 750 步，教训：RNG 上线前必须先过分布测试）。实测：权重 1.4s，prefill 0.09s，AR 41ms/步，e2e 文本 12 步停（torch 13 步）1.92s 音频共 2.2s；同 seed 按位确定；中文同 pathway 过。完整波形不对 torch 比（x0 不同即混沌分叉，M15 已证）——正确性由 teacher-forced 环担保，CLI 只担保确定性 + 分布 + stop  sane。

## 9. 回灌 scaled/raw 大 bug（M19，用户听出后段糊报的案）
症状：CLI 中文“你好清晰、后面消融成杂音”，能量包络正常。teacher-forced 全过、stop 全对——能掩盖 free-run 回灌 bug 的测试全是绿的。
根因：`generation.py:1022` 回灌用的是 RAW `pred_latent`，我喂了 scaled（`/0.4127`≈2.4 倍），每步复利 + teacher-forced 从不跑回灌 + 真环没给 loc 加门限 = 三重漏网。
修复：CLI + 两真环改喂 RAW；真环加 loc 门限（<1.0：scaled 版 err ~5，正常版落 bf16 地板 5e-2，判别力足够）。
效果：真环 drift-z 11.5→0.186（60x），loc 正中 bf16 地板。中文 CLI 12 步 1.92s，能量形态健康，`/tmp/opencode/cli_zh2.wav` 待听。
教训（§6 补第 8 条）：回灌/cond/scale 三处 latent 形态必须逐处对源码点名，测试必须给回灌加门限——stop 对不代表音频对。

## 10. VAE encoder（M20，本轮，clone 纯 Rust 前置）

`src/vae_enc.rs`：encoder 121 tensors 中 118 个（`fc_logvar` 推理无用，sigma 后验只取 mode——断言 shapes，跳过有注释）；res 全 depthwise k7 + pointwise（已有 causal 核直用）；4 个 strided dense conv 走 im2col+sgemm（torch floor 语义，hop 对齐下值精确）；fc_mu k3。`tests/vae_enc.rs`：exact-multiple/non-multiple pad/真 reference 四条，worst 5.9e-5（门限 1e-3，17x，按铁律不用动）。性能：ref（46 帧）1.14s vs torch 0.55s——瓶颈是末级 strided（67M 参数，49G MACs），一次性成本（reference 可缓存），认了。下一步：clone CLI（resample + speech segment fusion + ECAPA/VAE-enc/DiT 全串）。

## 11. Resample（M21，本轮，clone 纯 Rust 前置之二）

`src/resample.rs`：逐字复刻 torchaudio sinc_interp_hann（width=6/rolloff=0.99；kernel f64 算完转 fp32；零 padding左width/右width+orig；polyphase stride；ceil 目标长度；同率 passthrough）。8 组变采样率（44.1k/48k/22.05k/24k/16k 互转）一次过，worst 4.2e-6（门限按铁律取 5e-5，12x）。毫秒级，一次性成本。下一步：clone CLI（speech segment fusion + 全串）。

## 12. Clone CLI（M22，本轮）

`cutetts --mode voice_clone`：hound 读 wav → mono → 30s 截断 → reference/speaker 双分支（整段重复补 2s，顺序与 torch 一致：先截断再 resample）→ ECAPA + VAE-enc + `clone_prefix`（5 段融合，speaker slot 经 lm_speaker_linear）→ 同 AR 环（spk=Some）→ wav。`tests/prefix.rs`：piece 精确、ref feats 5.5e-5、文本行按位、speech 行 4.7e-2（bf16 地板）。实测：13 步 2.08s，总 ~4s（weights 1.4 + reference 1.8 + prefill 0.18 + AR 0.6 + VAE 0.2）。虚惊：prefill T=65 显示 3.4s——计时点框错（含 weights+reference），真值 0.13s。

## 13. 长文本分包（M23，本轮）
- 定性：小 distill LM 吃不下 1000+ token prefix——官方 torch 同文本同 seed 跑出 120s（750 步顶满），后半同样变外语；非 port bug（T=1296 teacher-forced prefill maxerr 4.42e-4，地板）。
- 对照：better-tail-HNR.wav（QORA 1.7B）277.6s ≈ 6 字/s 正常语速；我们单 prefix 36.6s ≈ 46 字/s 且跳段。
- 做法：移植 QORA `chunk.rs`（split_sentences 中英标点+小数/Mr. 守卫+150 字硬切，pad_tail 0.15s，crossfade 720=30ms）→ `src/chunk.rs`；CLI `--text-file` 多句自动分包，chunk seed=base+idx，单句路径无 pad 保持 bit-identical（--seed 12345 replay 验证）。
- 门：`tests/chunk.rs` 14 项（含 result.txt 真实文本无损切分）；result.txt→39 chunks/1618 步/263s（RTF~0.44），与 277s 参考同量级。

## 14. 分包三修（M24，本轮听感反馈）
- 包尾长静音（"总结与建议"前久等）：只 pad 不 trim，尾静音原样保留。移植 QORA
  `trim_silence`（内部/尾部静音压到 0.25s）→ trim→pad(0.15)→crossfade(720)。
- 逐句音色漂移：chunk seed=base+idx 使每包 x0 流不同。加 `--chunk-seeds same|incr`，
  默认 same（全包同一种子，音色稳定；复现仍确定）。
- 数字乱读：QORA 无 TN，新写 `src/tn.rs`（整数万亿分组/小数点/百分之/4位+年逐位/
  ~/-取到/负号/度乘除以），先归一化再分包，`--no-tn` 可关。
- 门：`tests/chunk.rs` 20 项，`tests/tn.rs` 3 项；result.txt v2：39 包/262.6s，
  单句路径仍 bit-identical。待耳朵验收。

## 15. 流式输出（M25，本轮）
- `--stream`：每 AR 步 2 帧 latent 直喂 `StreamingDecoder`（M17），PCM 边生成边
  追加写 wav（header 预写、结束回填），首包 = prefill + 1 步（短句实测 0.12s）。
- 流式 vs 离线差 3.05e-5 ≈ 1 LSB（int16 量化地板，不可闻）；多包流式为 raw
  concat（trim/pad/crossfade 跳过，测试模式已在 help 注明）。
- 门：`wav_stream_roundtrip`（头回填+i16 限幅）；短句流式 e2e 首包+对差。

## 16. 管道播放（M26，本轮）
- `--output -`：裸 s16le 单声道 24k 到 stdout；`say!` 宏让人话日志让路 stderr，
  管道保持干净。`StdoutRaw` 下游关闭（head）时静默 exit(0)，不 panic。
- 门：管道字节 == 文件模式 wav 体（bit-identical）；head 截断干净退出。
- 播法：`... --output - --stream | play -t raw -r 24000 -e signed -b 16 -c 1 -`
  或 `... | ffplay -f s16le -ar 24000 -ac 1 -i -`。

## 17. 流式实时性仪表（M27，本轮）
- 三个数：首包（prefill+1 步，短句 0.16s）、步进 pacing（每包 160ms 音频预算，
  实测 mean 91ms p99 108ms：含逐包 VAE，比离线 AR 纯环慢是单包无 batching 的代价）、
 整体 RTF（短句 0.60，含 prefill）。

## 18. 包响度拉平（M28，本轮听感反馈：克隆"一下大一下小"）
- 根因：每包是独立 AR 轨迹，响度天然漂移（实测 39 包增益 0.63~1.74，差 2.8 倍）；
  官方 `_normalize_decoded_waveforms` 只是 squeeze、无响度归一，单跑盖住了问题。
- 修：`level_chunks`（语音门限 RMS→中位数锚点，非零中位数，增益夹 [0.25,4]，
  静音包不动）→ pad → xfade → 全局 peak 0.98；`--no-level` 可关。
- 数：result.txt 逐秒 RMS std 0.051→0.022（-57%），max 0.33→0.18。

## 19. TN v2：型号读法（M29，--print-chunks 审计发现）
- 事故：Qwen3→Qwen三（应逐位）、H100→H一百、x86→x八十六、AVX-512→AVX负五百一十二、
  RTX 4090→RTX四千零九十、"2020 年"年份带空格未识别。
- 规则：贴字母的整数逐位（Qwen三/H一零零/x八六/RTX四零九零），单位后缀保持分组
  （八十GB/四十dB/一点七B），字母-数字连字符丢弃（AVX五一二），版本范围连读
  （x86-64→八六六四），年份容忍空格，字母+空格+数字视同型号。
- 门：tests/tn.rs 5 项；`--print-chunks` 审计 39 包干净；v5 全量 260.6s。

## 20. Base CFG wired, tts (P1，本轮）
- base=distill 减 6 个 embedding key，其余 163 key 同 shape/dtype；loader 55|61 兼容。
- `euler_sample_cfg`：sway 网格（10 步，coeff -0.8）+ 双分支 v=vc+2*(vc-vu)，
  独立 cond/uncond 历史，cond 分支 stop，同 RAW 反馈喂双分支。
- 门：sway 网格 1e-6（端点精确），base _predict 1e-4，euler-CFG replay 5e-4，
  teacher-forced DiT-CFG 1.69e-5/wav 6.5e-6/stop 13-13，真环 loc 4.2e-5。
- CLI：config.json variant 自适应 + `--cfg-strength/--sway/--diffusion-steps`
 （torch 同名同默认值同校验）；base voice_clone 明确拒绝（下轮）。
- 代价：同文本 distill 13 步 RTF~0.4 → base 12 步 RTF 1.32（10 步×双分支+2×LM）。
  A/B 对：base_en42.wav（我们 1.92s）vs torch_base_en42.wav（2.08s）。

## 21. P2 突发自动重试（M30，本轮听感：123 的洞 / 42 的切尾）
- 剖面：s123 第 5-6 包 0.00（洞），en42 尾包 0.08 硬切，s7 尾部 0.03→0.02 自然收敛。
- `stability()` 门：洞（末两包之前任一包均值 < 0.005）+ 尾部不收敛（尾包 > 0.05），
  阈值 `--hole-thresh/--tail-thresh` 可调；`--retries N` 逐包 seed+attempt 重试，
  首个通过即收留，全挂保留末次；retries=0 保持 bit-identical；与 --stream 互斥。
- 验证：seed 123 attempt0 Hole(5) 拒收 → seed 124 收敛 0.02→0.01 收留。

## 22. Base voice_clone wired (P1 continued)
- Uncond branch in clone is suffix-only (plan.unconditional has no reference);
  DiT uncond speaker row is zeros through bias-less adaln (exact zeros, NOT
  the plain path) — replicated op-for-op with explicit zero adalns in
  euler_sample_cfg (spk=None keeps both branches plain, tts behavior kept).
- Stability gate fix: Hole now requires resumed speech after the silence
  (pure trailing runs = stop lag, trim's job). First base-clone smoke run
  false-fired Hole(9) on an 11-patch trailing run; fixed + regression test.
- Gates: base _predict+adaln 6.1e-6, full euler-CFG-speaker 1.1e-5, clone
  teacher-forced DiT-CFG 2.4e-4 / wav 1.8e-5 / stops 60-60, true-ring loc
  3.2e-2. CLI smoke: base clone 20 steps, 3.20s, RTF 1.14.

## 23. 单包默认 trim + 性能方法论对齐 QORA（M31）
- 单包路径此前为保 bit-identical 而写裸输出，base clone 实测 3.2s 音频含
  1.9s 尾部静音（stop 滞后）。改默认 trim（0.25s）+ peak 0.98，
  `--no-trim` 恢复旧行为。验证：3.20s → 1.60s。
- 性能方法论（QORA-TTS-12Hz-1.7B docs/avx2-plan, decoder-q4-plan, pmu-pitfall）：
  门禁0先量后动、解析流量模型优先于 PMU（LLC-miss 漏 prefetch 流量 10 倍+）、
  unsafe 单点包络+标量 oracle、不达标杀项目。

## 24. 门禁0：base 单步流量模型（未动手，先量）
- 权重（fp32 常驻，bf16 load 时展开）：LM 508MB × 双分支 ≈ 1.0GB，
  DiT 282MB × 20 predicts ≈ 5.6GB，LocEnc ≈ 0.1GB → **≈6.7GB/AR 步**。
- 210ms/步实测 ⟹ ~32GB/s，带宽墙；DiT 占流量 84%，distill 复算
  1.6GB/步 ≈ 50ms 与实测 40ms 吻合，模型可信。
- 结论：性能工作从 DiT 开始。Stage 1 = bf16-DiT（fp32 累加，流量减半，
  预期步进 -40%）；Stage 2 = int8/Q4（待 Stage 1 门禁过后再议）。
  杀线：Stage 1 必须实测 RTF 增益 + 耳验 A/B 通过，否则关闭。

## 25. P3 WER 链路（M32：工具链通，基线见下）
- ASR：FunASR paraformer-zh（CPU，uv + tuna 源安装；modelscope 拉模型通；
  file-path 输入在其 audio loader 上 SIGSEGV，改走 numpy 桥接；24k→16k scipy）。
- 基线（2026-10-02）：en-short ×2 WER 0.000；zh-num-decimal CER 0.000；
  zh-num-year-pct CER 0.000；zh-num-model CER 0.150（3 错全在 "Qwen" 拉丁首字母，
  ASR 弱项非 TTS——paraformer 对生僻拉丁转写成汉字，RTF/H 等常见缩写可辨）。
- 结论：TN 数字链（小数/年份/百分/量词）在中文语境下端到端全对；
  拉丁型号词（Qwen/H100/RTX）超出本 ASR 测量能力，继续用耳朵仲裁。
  回归集：scripts/wer_manifest.json（5 case：文本/seed/ref_tn 锁定）。

## 26. Base A/B 耳验通过（2026-10-02）
- 对照：base_s7.wav（我们，13 步 2.08s）vs torch_base_en42.wav（13 步 2.08s），
  另附 base_en42.wav（12 步吞尾）作反例参照。
- 结论：清晰度 ok、自然度 ok、无怪音。base tts 线可交付。
