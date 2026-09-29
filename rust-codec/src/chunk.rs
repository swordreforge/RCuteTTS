//! Long-text chunking: split text into sentences, crossfade-concat audio.
//!
//! Ported from QORA-TTS-12Hz-1.7B `src/chunk.rs` (same repo, battle-tested
//! there incl. its unit tests, ported to `tests/chunk.rs`). Why: the small
//! distill LM cannot hold a 1000+ token prefix — official torch free-runs
//! wander off into gibberish on result.txt (1682 chars), while per-sentence
//! chunks stay in-distribution. QORA's 1.7B does the same for long inputs.
//!
//! Split boundaries: CJK 。！？；… newline, ASCII ! ? ; newline, and `.`
//! followed by whitespace or end (avoids splitting decimals like 3.14).
//! Over-long sentences (>150 chars) are hard-split on ，,、；.

/// Split text into non-empty sentence chunks.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        cur.push(c);
        let hard = matches!(c, '。' | '！' | '？' | '；' | '…' | '\n' | '!' | '?' | ';');
        let mut soft = false;
        if c == '.' && !chars[i].is_ascii_control() {
            // '.' splits only when followed by whitespace/end AND preceded by
            // CJK or when it looks like a sentence end (not a decimal/initial).
            let next_boundary = i + 1 >= chars.len() || chars[i + 1].is_whitespace();
            let prev = if i > 0 { chars[i - 1] } else { ' ' };
            let prev_is_digit = prev.is_ascii_digit();
            let next_is_digit = i + 1 < chars.len() && chars[i + 1].is_ascii_digit();
            soft = next_boundary && !(prev_is_digit && next_is_digit);
            // "Mr." / "Dr." guard: short alpha word before the dot
            if soft && prev.is_ascii_alphabetic() {
                let mut j = i;
                while j > 0 && chars[j - 1].is_ascii_alphabetic() {
                    j -= 1;
                }
                if i - j <= 2 {
                    soft = false;
                }
            }
        }
        if hard || soft {
            // absorb closing quotes/brackets right after the boundary
            while i + 1 < chars.len() && matches!(chars[i + 1], '"' | '\'' | '”' | '’' | '」' | '』' | '）' | ')') {
                i += 1;
                cur.push(chars[i]);
            }
            let s = cur.trim().to_string();
            if !s.is_empty() {
                out.push(s);
            }
            cur = String::new();
        }
        i += 1;
    }
    let s = cur.trim().to_string();
    if !s.is_empty() {
        out.push(s);
    }
    // hard-split over-long chunks on inner commas
    let mut final_out = Vec::new();
    for s in out {
        if s.chars().count() > 150 {
            final_out.extend(split_long(&s));
        } else {
            final_out.push(s);
        }
    }
    final_out
}

fn split_long(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        cur.push(c);
        if matches!(c, '，' | ',' | '、' | '；' | ';') && cur.chars().count() >= 40 {
            out.push(cur.trim().to_string());
            cur = String::new();
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// Ensure at least `min_tail_secs` of trailing silence (digital zeros).
/// Fixes hot-hot seams: a chunk ending in speech gets a clean stop + pause
/// so the join crossfade blends silence into the next attack instead of
/// mixing two phonemes. Bit-preserving when the tail already qualifies
/// (pure append, never touches existing samples).
pub fn pad_tail(audio: &[f32], sample_rate: u32, min_tail_secs: f32) -> Vec<f32> {
    if audio.is_empty() || min_tail_secs <= 0.0 {
        return audio.to_vec();
    }
    let frame = (sample_rate as usize / 50).max(1);
    // measure existing trailing silence (20ms frames, -40dB, same convention)
    let n_frames = audio.len().div_ceil(frame);
    let mut sil_frames = 0usize;
    for i in (0..n_frames).rev() {
        let end = ((i + 1) * frame).min(audio.len());
        let seg = &audio[i * frame..end];
        let e: f32 = seg.iter().map(|v| v * v).sum::<f32>() / seg.len() as f32;
        if e < 1e-4 {
            sil_frames += 1;
        } else {
            break;
        }
    }
    let have_secs = sil_frames as f32 * frame as f32 / sample_rate as f32;
    if have_secs >= min_tail_secs {
        return audio.to_vec();
    }
    // sample-exact deficit (round, not ceil: avoids f32 0.15*24000=3600.0001 → 3601)
    let target = (min_tail_secs * sample_rate as f32).round() as usize;
    let have_samples = (sil_frames * frame).min(audio.len());
    let need = target.saturating_sub(have_samples);
    if need == 0 {
        return audio.to_vec();
    }
    let mut out = audio.to_vec();
    out.extend(std::iter::repeat(0.0).take(need));
    out
}

/// Output len = sum - fade_len * (n-1). fade_len clamped to shortest chunk.
/// Concatenate chunks with a linear crossfade of `fade_len` samples.
pub fn crossfade_concat(chunks: &[Vec<f32>], fade_len: usize) -> Vec<f32> {
    if chunks.is_empty() {
        return Vec::new();
    }
    if chunks.len() == 1 {
        return chunks[0].clone();
    }
    let min_len = chunks.iter().map(|c| c.len()).min().unwrap_or(0);
    let fade = fade_len.min(min_len.saturating_sub(1).max(0));
    if fade == 0 {
        return chunks.concat();
    }
    let total: usize = chunks.iter().map(|c| c.len()).sum();
    let mut out = Vec::with_capacity(total - fade * (chunks.len() - 1));
    out.extend_from_slice(&chunks[0][..chunks[0].len() - fade]);
    for (idx, pair) in chunks.windows(2).enumerate() {
        let (a_tail, b) = (&pair[0][pair[0].len() - fade..], &pair[1]);
        for k in 0..fade {
            let t = k as f32 / fade as f32;
            out.push(a_tail[k] * (1.0 - t) + b[k] * t);
        }
        let rest_end = if idx + 1 == chunks.len() - 1 { b.len() } else { b.len() - fade };
        out.extend_from_slice(&b[fade..rest_end]);
    }
    out
}
