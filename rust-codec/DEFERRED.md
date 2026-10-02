# DEFERRED：决议不做 / 暂缓项（先存档，不动手）

> 规矩：每项必须有理由 + 重开条件。重开 = 先过门禁0，不直接写代码。
> 更新日期：2026-10-02。

## 冻结（有数据，不做）

1. **prefill 微优化**（0.061 vs 0.040 @T=48）
   理由：DRAM 带宽墙（fp32 vs torch bf16），占总量 2~3%。
   重开：首包 <100ms 变硬指标。
2. **bf16-DiT Stage 1**：已 kill（PERF_PLAN §7，实测更慢）。
   重开：AMX-BF16 tile 核，或换机器重测 gate-4。
3. **clippy style  pass**（unsafe 块多余括号等）：零行为收益，碰热代码。
   重开：单独立项，不搭车。

## 暂缓（等前置结论）

4. **int8/Q4 DiT Stage 2**：等门禁0重跑（融合后数字变了）+ 立项批准。
5. **外接 enhancement 模型**：等 EQ 版 + base 长文耳验 verdict；
   通过则按 PERF_PLAN 门禁 1~4 立项（默认关，f32 主链不动）。
6. **跨包韵律记忆**：动 AR 架构，研究级；等 leveling/耳验反馈证明非做不可。
7. **WER 拉丁覆盖**（whisper EN ASR）：TN 拉丁规则迭代时再议，目前耳朵仲裁。
8. **单包 `--no-trim` 遗留行为**：保留 flag，无动作。

## 否决（有关闭依据，不重开）

9. **iGPU / 外部推理库 / KV-cache 量化**：QORA 同结论（cache 小、无意义）。
10. **Windows/ARM 移植**：x86_64 AVX2 基线，非 x86 走 scalar 回退。
11. **decoder-Q4 式"先量权重占比再动手"豁免**：任何性能项目无门禁0不得开工。
