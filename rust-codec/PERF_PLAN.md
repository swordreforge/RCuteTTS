# CuteTTS 性能优化计划书（QORA 方法论）

> 地位：P1 功能线（base tts/clone）已交付，门禁全绿；性能是下一条独立线。
> 方法论对齐 `../QORA-TTS-12Hz-1.7B/docs/`（avx2-plan / decoder-q4-plan /
> pmu-pitfall）：**门禁0先量后动、解析流量模型优先于 PMU、unsafe 单点包络、
> 不达标杀项目**。本计划只含测量与路线，不含代码改动；每个 Stage 独立拍板。

---

## 1. 事实基线（已测，不猜）

### 1.1 实测耗时（22 线程，同机）

| 场景 | prefill | AR 步进 | 整体 RTF |
|---|---|---|---|
| distill 短句 | 0.07–0.20 s（T=23~26） | ~68 ms/步 | ~0.4 |
| distill 长文 39 包 | — | — | 0.44 |
| base 短句（12 步） | 0.12–0.18 s（T=23） | ~210 ms/步 | 1.32 |
| base 流式管 | — | step mean 91 ms（distill） | 1.00 |

Stage 0 phase-split 实测（`--profile-steps`，"Hello world..."，seed 42）：

| phase | base（12 步） | distill（13 步） |
|---|---|---|
| DiT euler | mean 159.8ms **(77%)** | mean 33.2ms (49%) |
| LocEnc 反馈 | 3.6ms (2%) | 3.6ms (5%) |
| LM decode（双分支/base 单分支） | 20.3ms (10%) | 10.1ms (15%) |
| VAE（offline 摊销） | 24.1ms/步 (12%) | 21.6ms/步 (32%) |

- base DiT 占比 77%，落进 80%±10% 门 → **解析模型确认，动手只动 DiT**。
- distill 步进里 VAE 摊销占 1/3（整句解码内存搬运），但绝对值小；
  Stage 1（bf16-DiT）对两者都有效，distill 绝对收益小。
- 附带：同文本 seed 42 在 stability 门下分别报 AbruptEnd(0.08) /
  Hole(6)——属 P2 抽签范畴，与本计划正交。

- base euler_sample_cfg 全采样：0.23–0.25 s/次（含 20 predicts）→ DiT 占主导。
- torch 对照：prefill T=48 我方 0.061 vs torch 0.040（已冻结，见 §5）。

### 1.2 门禁0：单步流量模型（解析法，PLAN §24）

权重 fp32 常驻（bf16  checkpoint 在 load 时展开，无运行时复算）：

| 分量（base） | 常驻字节 | 每步 pass 数 | 每步流量 |
|---|---|---|---|
| LM（Qwen 127M） | 508 MB | 双分支 ×1 | ~1.0 GB |
| DiT（70.5M） | 282 MB | 10 步 × 双分支 = 20 | **~5.6 GB（84%）** |
| LocEnc | ~0.1 GB | ×1 | ~0.1 GB |
| 合计 | — | — | **~6.7 GB/步** |

210 ms/步 ⟹ ~32 GB/s，带宽墙。distill 复算 1.6 GB/步 ≈ 50 ms，
实测 40 ms——模型可信（误差来自 pack 对齐填充与 prefetch）。

**结论：性能工作从 DiT 开始，别处（prefill、VAE、线程数）都没有结构性 headroom。**

### 1.3 PMU 使用规矩（抄 pmu-pitfall）

- `LLC-misses` 只看数量级 + 同形状比较；权重 streaming 被 prefetcher
  搬运，不计数，绝对值少 10 倍以上——禁止用它算"权重占比"。
- 要总量：先解析（§1.2 表），再用 demand-miss 做下界交叉验证。
- 无 root：不碰 uncore；multiplexing 缩放只看趋势，要精确就钉核减事件。

---

## 2. 不变量（动不得）

- fp32 全精度路径永不删除：`--dit-prec f32` 永远可用，当质量仲裁者。
- 默认精度：耳验通过前永远是 f32；通过后才翻默认。
- pure-Rust：不引 C/C++ 量化库（QORA 的 AGPL 红线同样适用）。
- 门禁体系：PLAN §6 铁律延续——teacher-forced 严格门只许收紧不许放松；
  量化 by design 改变数值，验收走**质量门禁**（§4），不走比特门。

---

## 3. 分阶段设计

### Stage 0：phase-split 仪表（无风险，先行）

- CLI 每步拆 LM/DiT/VAE/LocEnc 四段计时（`--profile-steps`），输出
  p50/p99。目的：把 §1.2 的解析模型变成实测分解，Stage 1 前后对比用。
- 门禁：数字与 §1.2 自洽（DiT 占比 80%±10%），否则先修模型再动手。

### Stage 1：bf16-DiT（主菜，低风险探路）

- DiT 权重 fp32→bf16 常驻（`half` 已在依赖），GEMV/GEMM 输入 bf16、
  **fp32 累加**（只切权重流量一半，计算语义不动）。
- 开关：`--dit-prec f32|bf16`，默认 f32。
- 预期：DiT 流量 5.6→2.8 GB，base 步进 -35~40%（RTF 1.32→~0.9）；
  distill 步进 -25%（RTF 0.4→~0.3）。
- 工作量：bf16 GEMV 核（复用现有 AVX2 SAXPY 风格，先标量正确版，
  再向量化）+ 打包格式（pack_a 的 bf16 版，或即时转换——先量再选）+
  全套门禁（§4）。

### Stage 2：int8/Q4 DiT（正餐，待定）

- 仅当 Stage 1 全部门禁通过才立项。DiT transformer q/k/v/o + MLP
  量化到 group-32 Q4（QORA talker 系同布局：16 packed bytes + f16
  scale，反量化 `(q-8)*scale`），norm/bias/adaln 保持 fp32，
  codebook 不适用（DiT 无 codebook）。
- 上限估算：DiT 流量再减半，base 步进 → ~0.12 s（RTF ~0.75）。
- 风险：Euler 10 步 × CFG 差分会放大误差（`vc-vu` 是差分结构，
  对量化噪声敏感）——门禁 2 强制覆盖长块 + clone adaln 路径。

### 备选（不立项，记录结论）

- 线程钉 P 核：taskset 实测过再说，预期个位数 %（带宽墙面前没用）。
- prefill 微优化：已冻结（PLAN §24，2~3% 总量）。
- VAE 整解码→流式默认化：VAE 占比 <5%，且 stream.rs 已有，不碰默认路径。
- KV cache 量化：cache 小，无意义（QORA 同结论）。
- iGPU / 外部推理库：否决（QORA 同结论）。

---

## 4. 质量门禁（任一不过即回退，不合并）

> 量化 by design 改变数值，验收全部走质量门禁，不走 sha。

- **门禁 0（动手前）**：本计划 §1 即门禁 0。Stage 2 立项前重跑一次
  （Stage 1 落地后数字会变）。
- **门禁 1（客观差分）**：teacher-forced 全套重跑——`tests/dit_base.rs`
  的 predict/euler 门从比特级放宽到 `5e-3`（bf16 量级；int8 另议），
  `e2e_base*` wav 门放宽到 `1e-2`；stop 序列必须 **13/13、60/60 不变**
  （stop 错一位即回退——吞尾不可接受）。
- **门禁 2（主观）**：用户耳验 A/B（f32 vs 新精度，同文本同 seed，
  短句 + result.txt 长文 + clone 各至少一条）。出现可闻底噪/金属音/
  音色漂移即回退。
- **门禁 3（回归）**：全量现有门禁通过（精度相关门按门禁 1 的放宽值，
  其余不变）；`--dit-prec f32` 路径输出与立项前逐字节一致。
- **门禁 4（性能）**：固定基准（result.txt，seed 67，22 线程）实测 RTF
  提升（Stage 1：base ≥25%），否则不合入。

---

## 5. 实施步骤（拍板后按序执行）

1. Stage 0：`--profile-steps` + 数字回填 §1（确认 DiT 占比）。
2. Stage 1：bf16 打包/核 + `--dit-prec` 开关 + 门禁 1~4。
3. 耳验 Stage 1（门禁 2）。过 → 默认 bf16（f32 留仲裁）→ 写实测数回本文档；
   不过 → 关闭 Stage 1，整条线止步（Stage 2 不立项）。
4. Stage 2：重跑门禁 0 → 立项评审 → Q4 DiT + 全门禁 + 终验。
5. 每步提交独立可回退（flag 关闭即回 f32，零耦合）。

## 6. 非目标

- LM/VAE/LocEnc 精度（Stage 1 后看占比再议，不预立项）。
- 跨包韵律记忆、base 长文分包策略（功能线，与本计划正交）。
- Windows/ARM 移植（x86_64 AVX2 基线不变，非 x86 走 scalar 回退）。

## 7. Stage 1  kill 记录（in-kernel bf16，门禁4 未过，关闭）

- 实现过：bf16 panels + in-loop zero-extend（micro_8x8_bf16，permutevar
  splat）+ GEMV 路径（run_gemv8b）+ `PackedW` enum + `--dit-prec`，
  约 620 行，未合入即切除（Fehlschlag dokumentiert, Code entfernt）。
- 实测（base 短句同 seed）：
  - fp32 micro：DiT 159.8 ms/sample；
  - bf16 micro：202.6 ms（**更慢**）；
  - bf16 GEMV 路径（t≤8 改道）：508 ms（面板 8 次重读，大败）。
- 根因：8x8 outer-product 要求一侧 splat；fp32 用内存 broadcast
  （load 端口），bf16 须先转寄存器再 permute splat（shuffle 端口串行，
  8 cycles/ii 打底，FMA 端口挨饿）。P 核钉死复测依然更慢——非 E 核锅。
  门禁0 的"零 core 开销"假设错了，流量减半的 87 ms 被 shuffle 多出的
  ~120 ms 吃掉还倒贴。
- 质量侧（供参考，不作为关闭依据）：sgemm 相对 1.4e-3，predict 3.9e-3，
  4 步 euler 5.8e-3，10 步 CFG（含 `vc-vu` ×3 放大）2.3e-2；
  teacher-forced wav RMS 1.2~1.7e-3（-37dB），stop 序列全程精确——
  **慢是唯一的死因**，数值本可接受。
- 重开本路线的唯一路径：AMX-BF16 tile 核（155H 有 AMX，1 TFLOPS+ 级，
  但 tile 配置/信号抢占/回退链是 Stage-2 量级工程），或换一台
  shuffle 不瓶颈的机器重测 gate-4。

## 8. 新 gate-0 候选：CFG 双分支面板融合（未动手，先量）

- 动机：DiT 20 predicts/步里 cond/uncond 分支是同一权重先后各扫一遍
  （5.6GB 中的一半是重复流量）；面板级融合（同一 panel 一次读出、
  双分支各算各的）可省 ~40% DiT 流量，**不碰精度**（每分支 op 序不变
  → 可逐位验证）。
- 动手前先量：确认双分支权重重读确实各走一次 DRAM（解析上是，perf
  抽查 cache-miss 趋势），再估融合后的 panel 常驻可行性。
- 杀线：融合版与现版逐位一致 + 实测步进 -25%，否则关闭。
