//! Text-normalization gates: integers, decimals, percents, years,
//! ranges, signs. No weights. Includes result.txt spot checks.
use cutetts_codec::tn::{int_zh, normalize};

#[test]
fn int_basics() {
    assert_eq!(int_zh(0), "零");
    assert_eq!(int_zh(5), "五");
    assert_eq!(int_zh(10), "十");
    assert_eq!(int_zh(14), "十四");
    assert_eq!(int_zh(97), "九十七");
    assert_eq!(int_zh(100), "一百");
    assert_eq!(int_zh(101), "一百零一");
    assert_eq!(int_zh(110), "一百一十");
    assert_eq!(int_zh(171), "一百七十一");
    assert_eq!(int_zh(1000), "一千");
    assert_eq!(int_zh(1010), "一千零一十");
    assert_eq!(int_zh(4090), "四千零九十");
    assert_eq!(int_zh(2572), "二千五百七十二");
    assert_eq!(int_zh(10000), "一万");
    assert_eq!(int_zh(12000), "一万二千");
    assert_eq!(int_zh(100000000), "一亿");
    assert_eq!(int_zh(10010), "一万零一十");
    assert_eq!(int_zh(100000), "十万");
}

#[test]
fn prose_cases() {
    assert_eq!(normalize("RTF为4.78"), "RTF为四点七八");
    assert_eq!(normalize("0.35"), "零点三五");
    assert_eq!(normalize("171毫秒"), "一百七十一毫秒");
    assert_eq!(normalize("2020年的笔记本"), "二零二零年的笔记本");
    assert_eq!(normalize("-40dB"), "负四十dB");
    assert_eq!(normalize("35%"), "百分之三十五");
    assert_eq!(normalize("0.7~0.9s"), "零点七到零点九s");
    assert_eq!(normalize("1.7B模型"), "一点七B模型");
    assert_eq!(normalize("2核2G"), "二核二G");
    assert_eq!(normalize("300毫秒内"), "三百毫秒内");
    assert_eq!(normalize("5.7秒"), "五点七秒");
    assert_eq!(normalize("0.5秒"), "零点五秒");
}

#[test]
fn model_names_digitwise() {
    assert_eq!(normalize("Qwen3-TTS"), "Qwen三-TTS");
    assert_eq!(normalize("H100"), "H一零零");
    assert_eq!(normalize("x86"), "x八六");
    assert_eq!(normalize("M1"), "M一");
    assert_eq!(normalize("int8"), "int八");
    assert_eq!(normalize("RTX4090"), "RTX四零九零");
    assert_eq!(normalize("AVX-512"), "AVX五一二");
    assert_eq!(normalize("x86-64"), "x八六六四");
    assert_eq!(normalize("1.7B模型"), "一点七B模型");
    assert_eq!(normalize("RTX 4090"), "RTX 四零九零");
    assert_eq!(normalize("171 毫秒"), "一百七十一 毫秒");
}

#[test]
fn year_with_space() {
    assert_eq!(normalize("2020 年的笔记本"), "二零二零年的笔记本");
}
#[test]
fn passthrough() {
    // pure Chinese prose is byte-identical
    let s = "今天天气真好，我们出去走走吧！";
    assert_eq!(normalize(s), s);
    // numeric dots become 点; the sentence-ending dot stays (splitter needs it)
    assert_eq!(normalize("Pi is 3.14. Nice."), "Pi is 三点一四. Nice.");
}
