"""WER/CER eval for CuteTTS Rust outputs (P3).
ASR: FunASR paraformer-zh (CPU). NOTE: pass wav as numpy (file-path input
segfaults in funasr 1.4.16's audio loader on this box); 24k -> 16k via
scipy. Metric is CER (Chinese) / WER (whitespace languages) on normalized
text: lowercase latin, full->half width, strip punct/space, keep CJK+alnum.

Reference must be what was SPOKEN: for --text-file runs that's the POST-TN
text (regenerate with `cutetts --print-chunks`, concatenated, or pass the
original file plus --tn to apply the same documented rules... simplest is
caller passes the exact spoken text file).

Usage:
  wer_eval.py hyp.wav ref.txt [--lang zh|en] [--sr 24000]
  wer_eval.py --manifest eval_manifest.json   # batch: [{wav, ref, lang, tag}]

Batch output: per-item CER + macro average. Exit 0 always (numbers, no gate).
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_16k(path: str, src_sr: int = 24000) -> np.ndarray:
    d, sr = sf.read(path, dtype="float32", always_2d=True)
    d = d.mean(axis=1)
    if sr != 16000:
        g = np.gcd(sr, 16000)
        d = resample_poly(d, 16000 // g, sr // g).astype(np.float32)
    return d


def norm(text: str, keep_space: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    out = []
    for c in text:
        if c.isspace():
            if keep_space:
                out.append(" ")
            continue
        if unicodedata.category(c).startswith("P"):
            continue
        if "a" <= c <= "z" or "0" <= c <= "9" or "\u4e00" <= c <= "\u9fff":
            out.append(c)
    return "".join(out)


def edit_dist(a: str, b: str) -> int:
    # char-level Levenshtein, O(min) memory
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> tuple[float, int, int]:
    r, h = norm(ref), norm(hyp)
    if not r:
        return (0.0 if not h else 1.0), 0, len(h)
    return edit_dist(r, h) / max(len(r), 1), len(r), edit_dist(r, h)


def cer_cjk_only(ref: str, hyp: str) -> tuple[float, int, int]:
    # strip latin/digit runs (ASR latin tax) — scores CJK body only
    r = re.sub(r"[a-z0-9]+", "", norm(ref))
    h = re.sub(r"[a-z0-9]+", "", norm(hyp))
    if not r:
        return (0.0 if not h else 1.0), 0, len(h)
    return edit_dist(r, h) / max(len(r), 1), len(r), edit_dist(r, h)


_asr = None

# FunASR degrades on multi-minute inputs: split into 30 s windows,
# transcribe each, concatenate (boundary artifacts negligible over 1k+ chars).
ASR_WIN = 30 * 16000


def transcribe(wav16: np.ndarray) -> str:
    global _asr
    if _asr is None:
        from funasr import AutoModel

        _asr = AutoModel(model="paraformer-zh", device="cpu", disable_update=True)
    if len(wav16) <= ASR_WIN:
        r = _asr.generate(input=wav16)
        return r[0]["text"] if r else ""
    parts = []
    for off in range(0, len(wav16), ASR_WIN):
        r = _asr.generate(input=wav16[off : off + ASR_WIN])
        parts.append(r[0]["text"] if r else "")
    return "".join(parts)


def eval_one(wav: str, ref_path: str, lang: str, src_sr: int, cjk_only: bool = False,
             hyp_cache: str | None = None, hyp_in: str | None = None) -> dict:
    ref = Path(ref_path).read_text().strip()
    if hyp_in:
        hyp = Path(hyp_in).read_text()
    else:
        hyp = transcribe(load_16k(wav, src_sr))
        if hyp_cache:
            Path(hyp_cache).write_text(hyp)
    if lang != "zh":
        rn = norm(ref, keep_space=True).split()
        hn = norm(hyp, keep_space=True).split()
        e = edit_dist(rn, hn)
        score, n, errs = e / max(len(rn), 1), len(rn), e
        unit = "WER"
    elif cjk_only:
        score, n, errs = cer_cjk_only(ref, hyp)
        unit = "CER-cjk"
    else:
        score, n, errs = cer(ref, hyp)
        unit = "CER"
    return {"wav": wav, "unit": unit, "score": score, "n": n, "errs": errs, "hyp": hyp}


def main() -> None:
    argv = sys.argv[1:]
    cjk_only = "--cjk-only" in argv
    argv = [a for a in argv if a != "--cjk-only"]
    hyp_cache = hyp_in = None
    args: list[str] = []
    skip = False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if a == "--hyp-cache":
            hyp_cache = argv[i + 1]
            skip = True
        elif a == "--hyp-in":
            hyp_in = argv[i + 1]
            skip = True
        else:
            args.append(a)
    if args and args[0] == "--manifest":
        items = json.loads(Path(args[1]).read_text())
        scores = []
        for it in items:
            r = eval_one(it["wav"], it["ref"], it.get("lang", "zh"), it.get("sr", 24000),
                         cjk_only or it.get("cjk_only", False))
            r["tag"] = it.get("tag", "")
            scores.append(r["score"])
            print(f"[{r['tag']}] {r['unit']}={r['score']:.3f} (errs={r['errs']}/{r['n']})")
            print(f"  hyp: {r['hyp'][:120]}")
        print(f"macro avg: {sum(scores) / max(len(scores), 1):.3f} over {len(scores)} items")
        return
    wav, ref = args[0], args[1]
    lang = "zh"
    sr = 24000
    for i, a in enumerate(args[2:]):
        if a == "--lang":
            lang = args[3 + i]
        if a == "--sr":
            sr = int(args[3 + i])
    r = eval_one(wav, ref, lang, sr, cjk_only, hyp_cache, hyp_in)
    print(f"{r['unit']}={r['score']:.3f} (errs={r['errs']}/{r['n']})")
    print(f"hyp: {r['hyp']}")


if __name__ == "__main__":
    main()
