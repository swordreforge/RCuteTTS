//! Chunking gates (ported from QORA-TTS-12Hz-1.7B `src/chunk.rs` tests).
//! Pure-logic, no weights. Adapted: our join pipeline is pad_tail-only
//! (no trim_silence port — pad alone guarantees the fade zone starts
//! from digital silence on the tail side).
use cutetts_codec::chunk::{crossfade_concat, pad_tail, split_sentences};

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
    println!("CHUNK gates passed");
}
