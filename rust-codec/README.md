# cutetts (Rust): pure-Rust CuteTTS-distill inference

Text → 24 kHz wav with no Python on the hot path. Parity-validated
module-by-module against torch (see `PLAN.md`), faster than torch on
decode/DiT, bit-deterministic given the same seed.

## Build

```bash
cd rust-codec
cargo build --release --bin cutetts
./target/release/cutetts --help
```

Weights + tokenizer come from `../model/CuteTTS-distill`
(override with `--model-dir`). Threads: `CUTETTS_THREADS` or `--threads`
(default: all logical CPUs).

## Recipes

```bash
# short tts
cutetts --text "你好，这是一个测试。" --output out.wav --seed 42

# voice clone (reference: any PCM wav, any rate/channels; capped at 30 s)
cutetts --mode voice_clone --reference-audio ref.wav \
  --text "换上这个音色说话。" --output clone.wav --seed 42

# long article: auto-splits into sentence chunks, crossfade join
cutetts --text-file article.txt --output long.wav --seed 67

# audit what the model actually sees (post-TN chunks + token counts)
cutetts --text-file article.txt --output /dev/null --print-chunks

# stream to file (first packet right after prefill + 1 step)
cutetts --text "边说边写。" --output s.wav --stream --seed 7

# generate and play (needs sox/ffplay/mpv on your machine)
cutetts --text "边说边播。" --output - --stream --seed 7 \
  | play -t raw -r 24000 -e signed -b 16 -c 1 -
cutetts --text-file article.txt --output - --seed 67 \
  | ffplay -f s16le -ar 24000 -ac 1 -i -
```

## Reproducibility (seed discipline)

Every run prints `seed <N> (user|random)`. Bit-identical replay needs
all three locked:

1. same binary, 2. same text file, 3. `--seed N` (log must say `(user)`).

Same seed is stable across thread counts. Omitted `--seed` draws a fresh
random seed each run (`seed 0` is remapped — xorshift guard).

## What the pipeline does

`--text-file` long input: text normalization → sentence split
(`src/chunk.rs`, QORA port) → per-chunk tokenize/prefill/AR/VAE
(chunk seed = base, `--chunk-seeds incr` for base+idx) → trim 0.25 s →
loudness leveling (speech-RMS to median, `--no-level` off) → 0.15 s tail
pad → 30 ms crossfade → peak 0.98. `--text` stays single-shot.
`--stream` emits raw PCM per AR step instead (no trim/level/crossfade).

Measured on this box (22 threads): prefill T=26 ≈ 0.07 s,
AR ≈ 40 ms/step offline (≈ 91 ms streamed, incl. per-packet VAE),
first packet ≈ 0.12–0.16 s, long article (39 chunks) RTF ≈ 0.4–0.6,
streamed pipe RTF ≈ 1.0.

## Known limits (won't-fix unless stated)

- Small LM (~0.13 B): 1000+ token prefixes wander **in official torch too**
  (teacher-forced prefill matches to 4.4e-4, so this is the model, not the
  port). Long text must go through `--text-file` chunking.
- Chunks are independent trajectories: same seed stabilizes timbre,
  leveling kills ±loudness, but prosody doesn't carry across chunks.
- Free-run length/content is chaotic (torch self-noise flips stop/length
  ~10×). Never expect two different seeds to agree; pick good seeds.
- Base model (`model/CuteTTS`, 10-step CFG) is not wired yet (distill only).
- Prefill micro-gap vs torch (0.061 vs 0.040 @T=48) frozen: DRAM-bandwidth
  wall (fp32 vs torch bf16), 2–3% of total.

## Tests

```bash
cargo test --release --bin cutetts   # rng/seed/arg/wav-stream unit tests
cargo test --release --test chunk     # split/trim/pad/level/crossfade gates
cargo test --release --test tn        # number-reading gates
cargo test --release --test prefix    # clone prefix parity (needs weights)
```

Differential-vs-torch fixtures live in `testdata/` (generators in
`scripts/`). Full milestone log and iron test rules: `PLAN.md`.
