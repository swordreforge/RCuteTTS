//! Chunking gates (ported from QORA-TTS-12Hz-1.7B `src/chunk.rs` tests).
//! Pure-logic, no weights. Adapted: our join pipeline is pad_tail-only
//! (no trim_silence port — pad alone guarantees the fade zone starts
//! from digital silence on the tail side).
use cutetts_codec::chunk::{crossfade_concat, level_chunks, pad_tail, peak_guard, speech_rms, split_sentences, trim_silence};

#[test]
fn split_chinese() {
    let v = split_sentences("今天天气真好，我们出去走走吧！你好吗？");
    assert_eq!(v, vec!["今天天气真好，我们出去走走吧！", "你好吗？"]);
}

#[test]
fn split_english() {
    let v = split_sentences("Good morning! How are you? I am well.");
    assert_eq!(v, vec!["Good morning!", "How are you?", "I am well."]);
}

#[test]
fn split_decimal_kept() {
    let v = split_sentences("Pi is 3.14. Nice.");
    assert_eq!(v.len(), 2);
    assert_eq!(v[0], "Pi is 3.14.");
}

#[test]
fn split_mr_guard() {
    let v = split_sentences("Mr. Smith is here. OK.");
    assert_eq!(v.len(), 2);
    assert_eq!(v[1], "OK.");
}

#[test]
fn split_newlines_and_empty() {
    let v = split_sentences("第一句。\n\n第二句。\n");
    assert_eq!(v, vec!["第一句。", "第二句。"]);
}

#[test]
fn split_no_punct() {
    assert_eq!(split_sentences("你好"), vec!["你好"]);
}

#[test]
fn split_long_hard_split() {
    let s: String = std::iter::repeat("测试，").take(80).collect();
    let v = split_sentences(&s);
    assert!(v.len() > 1);
    assert!(v.iter().all(|c| c.chars().count() <= 160));
    assert_eq!(v.concat(), s);
}

#[test]
fn split_result_txt_sane() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/result.txt"
    ))
    .unwrap();
    let v = split_sentences(text.trim());
    // 7 paragraphs of Chinese: expect sentence-scale chunks, none huge.
    assert!(v.len() >= 7, "got {} chunks", v.len());
    assert!(v.iter().all(|c| c.chars().count() <= 160), "all chunks short");
    // lossless: concat with no separator equals trimmed input minus
    // boundary whitespace (newlines are boundaries and get trimmed)
    let joined: String = v.concat();
    let stripped: String = text.trim().chars().filter(|c| !c.is_whitespace() || *c == ' ').collect();
    let joined_ns: String = joined.chars().filter(|c| !c.is_whitespace() || *c == ' ').collect();
    assert_eq!(joined_ns, stripped);
}

#[test]
fn crossfade_len_and_blend() {
    let y = crossfade_concat(&[vec![1.0; 1000], vec![0.0; 1000]], 100);
    assert_eq!(y.len(), 1900);
    assert_eq!(y[899], 1.0);
    assert!((y[900] - 1.0).abs() < 1e-6); // t=0 → a
    assert!((y[949] - 0.51).abs() < 0.02); // t=0.49
    assert!((y[999] - 0.01).abs() < 1e-5); // t=0.99
    assert_eq!(y[1000], 0.0);
}

#[test]
fn crossfade_three_chunks() {
    let y = crossfade_concat(&[vec![1.0; 500], vec![2.0; 500], vec![3.0; 500]], 50);
    assert_eq!(y.len(), 1400);
    assert!((y[0] - 1.0).abs() < 1e-6);
    assert!((y[1399] - 3.0).abs() < 1e-6);
}

#[test]
fn crossfade_edge_cases() {
    assert!(crossfade_concat(&[], 10).is_empty());
    assert_eq!(crossfade_concat(&[vec![1.0, 2.0]], 10), vec![1.0, 2.0]);
    let y = crossfade_concat(&[vec![1.0; 10], vec![0.0; 10]], 100);
    assert_eq!(y.len(), 11);
}

fn speech(len: usize, amp: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = i as f32 / 24000.0;
            amp * (0.6 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.4 * (2.0 * std::f32::consts::PI * 440.0 * t).sin())
        })
        .collect()
}

#[test]
fn pad_tail_noop_when_silent() {
    let mut a = speech(12000, 0.2);
    a.extend(vec![0.0; 4800]); // 0.2s trailing silence already
    assert_eq!(pad_tail(&a, 24000, 0.15), a);
}

#[test]
fn pad_tail_appends_when_hot() {
    let a = speech(12000, 0.2); // ends hot, zero trailing silence
    let p = pad_tail(&a, 24000, 0.15);
    assert_eq!(p.len(), 12000 + 3600);
    assert!(p[12000..].iter().all(|&v| v == 0.0));
    assert_eq!(&p[..12000], &a[..]);
}

#[test]
fn seam_pipeline_pad_only() {
    // worst case: both chunks hot at the seam; pad-only pipeline
    let a = speech(19200, 0.2);
    let b = speech(19200, 0.2);
    let pa = pad_tail(&a, 24000, 0.15);
    let joined = crossfade_concat(&[pa.clone(), b.clone()], 720);
    assert!(pa[pa.len() - 720..].iter().all(|&v| v == 0.0));
    assert_eq!(joined[pa.len() - 720], 0.0, "fade starts from exact silence");
    assert_eq!(joined.len(), pa.len() + b.len() - 720);
}

#[test]
fn trim_compresses_gap() {
    // speech + 1s silence (50 frames) + speech, max_gap 0.25s (13 frames ceil)
    let mut x = vec![0.1; 480];
    x.extend(vec![0.0; 480 * 50]);
    x.extend(vec![0.1; 480]);
    let y = trim_silence(&x, 24000, 0.25);
    assert_eq!(y.len(), 480 * 15, "len={}", y.len());
}

#[test]
fn trim_cuts_trailing() {
    let mut x = vec![0.1; 480];
    x.extend(vec![0.0; 480 * 50]);
    let y = trim_silence(&x, 24000, 0.25);
    assert_eq!(y.len(), 480 * 14, "len={}", y.len()); // 1 speech + 13 gap
}

#[test]
fn trim_short_gap_untouched() {
    let mut x = vec![0.1; 480];
    x.extend(vec![0.0; 480 * 10]);
    x.extend(vec![0.1; 480]);
    assert_eq!(trim_silence(&x, 24000, 0.25), x);
}

#[test]
fn trim_all_silence() {
    assert!(trim_silence(&vec![0.0; 480 * 5], 24000, 0.25).is_empty());
}

#[test]
fn trim_leading_kept() {
    let mut x = vec![0.0; 480 * 5];
    x.extend(vec![0.1; 480]);
    assert_eq!(trim_silence(&x, 24000, 0.25), x);
}

#[test]
fn trim_then_pad_bounds_tail() {
    // hot tail + 2s trailing silence: tail must shrink to [0.15, 0.25]s
    let mut a = speech(19200, 0.2);
    a.extend(vec![0.0; 48000]);
    let t = trim_silence(&a, 24000, 0.25);
    let p = pad_tail(&t, 24000, 0.15);
    let tail = p.len() - 19200;
    assert!(tail >= 3600 && tail <= 6000 + 480, "tail={tail}");
}

#[test]
fn level_equalizes_chunks() {
    // same content, 4x loudness apart → same speech RMS after leveling;
    // median anchor lands on the louder chunk here
    let a = speech(12000, 0.1);
    let b = speech(12000, 0.4);
    let (lv, gains) = level_chunks(&[a, b], 24000);
    assert!((gains[1] - 1.0).abs() < 1e-6, "median anchor untouched");
    let ra = speech_rms(&lv[0], 24000);
    let rb = speech_rms(&lv[1], 24000);
    assert!((ra - rb).abs() / rb < 1e-5, "{ra} vs {rb}");
    assert!((gains[0] - 4.0).abs() < 1e-4, "clamped gain");
}

#[test]
fn level_silence_safe() {
    // silence-only second chunk: gain stays 1.0 (never amplify silence
    // into noise), no NaN, content untouched
    let a = speech(12000, 0.2);
    let b = vec![0.0; 12000];
    let (lv, gains) = level_chunks(&[a, b], 24000);
    assert_eq!(gains[1], 1.0);
    assert!(lv[1].iter().all(|v| v.is_finite()));
    assert!(lv[1].iter().all(|&v| v == 0.0));
}

#[test]
fn peak_guard_caps() {
    let a = speech(12000, 2.0); // peaks > 1
    let peak_before = a.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(peak_before > 1.0);
    let g = peak_guard(&a, 0.98);
    let peak_after = g.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!((peak_after - 0.98).abs() < 1e-5);
    // under ceiling: bit-preserving
    let q = speech(12000, 0.2);
    assert_eq!(peak_guard(&q, 0.98), q);
    println!("CHUNK gates passed");
}
