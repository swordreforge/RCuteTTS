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

/// Compress internal silences longer than `max_gap_secs` down to `max_gap_secs`
/// and trim trailing silence beyond `max_gap_secs`.
/// Frame = 20ms, speech threshold = mean square energy >= 1e-4.
/// (QORA-TTS-12Hz-1.7B `src/wav.rs::trim_silence`, verbatim port.)
pub fn trim_silence(audio: &[f32], sample_rate: u32, max_gap_secs: f32) -> Vec<f32> {
    if audio.is_empty() || max_gap_secs <= 0.0 {
        return audio.to_vec();
    }
    let frame = (sample_rate as usize / 50).max(1);
    let max_gap_frames = ((max_gap_secs * sample_rate as f32) / frame as f32).ceil() as usize;

    // Classify frames
    let n_frames = audio.len().div_ceil(frame);
    let mut is_speech = vec![false; n_frames];
    for (i, s) in is_speech.iter_mut().enumerate() {
        let end = ((i + 1) * frame).min(audio.len());
        let seg = &audio[i * frame..end];
        let e: f32 = seg.iter().map(|v| v * v).sum::<f32>() / seg.len() as f32;
        *s = e >= 1e-4;
    }

    // Find last speech frame; drop everything after last_speech + max_gap_frames
    let Some(last_speech) = is_speech.iter().rposition(|&s| s) else {
        return Vec::new();
    };
    let keep_frames = (last_speech + 1 + max_gap_frames).min(n_frames);

    // Copy, skipping the middle of over-long internal gaps
    let mut out: Vec<f32> = Vec::with_capacity(audio.len());
    let mut i = 0;
    while i < keep_frames {
        if is_speech[i] {
            let end = ((i + 1) * frame).min(audio.len());
            out.extend_from_slice(&audio[i * frame..end]);
            i += 1;
        } else {
            let mut j = i;
            while j < keep_frames && !is_speech[j] {
                j += 1;
            }
            let gap = j - i;
            let keep = gap.min(max_gap_frames);
            for k in i..i + keep {
                let end = ((k + 1) * frame).min(audio.len());
                out.extend_from_slice(&audio[k * frame..end]);
            }
            i = j;
        }
    }
    out
}

/// Speech-gated RMS (20ms frames, 1e-4 threshold — same convention as
/// trim/pad): measures speech loudness, not pause content. Falls back to
/// plain RMS when no frame qualifies (avoids div-by-zero on silence).
pub fn speech_rms(audio: &[f32], sample_rate: u32) -> f32 {
    if audio.is_empty() {
        return 0.0;
    }
    let frame = (sample_rate as usize / 50).max(1);
    let n_frames = audio.len().div_ceil(frame);
    let mut num = 0.0f64;
    let mut den = 0usize;
    for i in 0..n_frames {
        let end = ((i + 1) * frame).min(audio.len());
        let seg = &audio[i * frame..end];
        let e: f32 = seg.iter().map(|v| v * v).sum::<f32>() / seg.len() as f32;
        if e >= 1e-4 {
            num += seg.iter().map(|v| (v * v) as f64).sum::<f64>();
            den += seg.len();
        }
    }
    if den == 0 {
        let n = audio.len();
        return (audio.iter().map(|v| (v * v) as f64).sum::<f64>() / n as f64).sqrt() as f32;
    }
    (num / den as f64).sqrt() as f32
}

/// Level multi-chunk loudness: scale every chunk's speech RMS to the
/// median across chunks (robust anchor — chunk 0 is often an outlier).
/// Medians over nonzero values only; all-silent input → gains of 1.
/// Gains clamped to [0.25, 4.0] so a silence-heavy chunk can't explode
/// into noise. Returns (leveled chunks, gains).
pub fn level_chunks(chunks: &[Vec<f32>], sample_rate: u32) -> (Vec<Vec<f32>>, Vec<f32>) {
    if chunks.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut rs: Vec<f32> = chunks.iter().map(|c| speech_rms(c, sample_rate)).filter(|&r| r > 1e-6).collect();
    rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let anchor = rs.get(rs.len() / 2).copied().unwrap_or(0.0);
    let mut out = Vec::with_capacity(chunks.len());
    let mut gains = Vec::with_capacity(chunks.len());
    for c in chunks {
        let r = speech_rms(c, sample_rate);
        let mut g = if r > 1e-6 && anchor > 1e-6 { anchor / r } else { 1.0 };
        g = g.clamp(0.25, 4.0);
        gains.push(g);
        out.push(c.iter().map(|v| v * g).collect());
    }
    (out, gains)
}

/// Global peak guard: scale down so peak <= `ceiling` (default 0.98).
/// Bit-preserving when already under the ceiling.
pub fn peak_guard(audio: &[f32], ceiling: f32) -> Vec<f32> {
    let peak = audio.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if peak <= ceiling || peak <= 0.0 {
        return audio.to_vec();
    }
    let g = ceiling / peak;
    audio.iter().map(|v| v * g).collect()
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
