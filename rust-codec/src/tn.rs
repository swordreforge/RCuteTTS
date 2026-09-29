//! Rule-based Chinese text normalization for TTS (numbers and units).
//!
//! The small LM mangles raw digits ("RTF为4.78", "2020年", "-40dB").
//! v1 scope is deliberately narrow — numeric tokens only; Latin prose and
//! code pass through untouched:
//! - integers → grouped reading (一百七十一, 四千零九十, 一万二千)
//! - decimals → 点 reading (四点七八), fractional digits read one by one
//! - trailing %/％ → 百分之X (百分之三十五)
//! - 4 digits + 年 → digit-by-digit (二零二零年)
//! - N~M / N～M / N-M (digits both sides) → N到M
//! - leading -/－ before digits → 负 (负四十)
//! - ° → 度, ×/*/÷ → 乘/乘/除以
//! Anything unrecognized passes through byte-identical.

const D: [&str; 10] = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"];

/// Integer reading with 万/亿/兆 grouping (0..=10^16-1; larger → digitwise).
pub fn int_zh(n: u64) -> String {
    if n == 0 {
        return "零".to_string();
    }
    if n >= 10_000_000_000_000_000 {
        return digitwise(&n.to_string());
    }
    let groups = ["", "万", "亿", "兆"];
    // split into 4-digit groups, low to high
    let mut parts: Vec<u16> = Vec::new();
    let mut v = n;
    while v > 0 {
        parts.push((v % 10000) as u16);
        v /= 10000;
    }
    let mut out = String::new();
    let mut need_zero = false; // a skipped zero-group (or group <1000) forces 零
    for (gi, &g) in parts.iter().enumerate().rev() {
        if g == 0 {
            need_zero = true;
            continue;
        }
        if !out.is_empty() {
            if need_zero || g < 1000 {
                out.push_str("零");
            }
        }
        need_zero = false;
        out.push_str(&group4(g));
        out.push_str(groups[gi]);
    }
    // leading 一十 → 十 ("十", not "一十"; but "一百一十" keeps inner 一十)
    if let Some(rest) = out.strip_prefix("一十") {
        out = format!("十{rest}");
    }
    out
}

/// 4-digit group 1..=9999, no group unit.
fn group4(g: u16) -> String {
    debug_assert!((1..=9999).contains(&g));
    let d = [(g / 1000) as usize, ((g / 100) % 10) as usize, ((g / 10) % 10) as usize, (g % 10) as usize];
    let u = ["千", "百", "十", ""];
    let mut out = String::new();
    let mut started = false;
    let mut zero = false;
    for (i, &x) in d.iter().enumerate() {
        if x == 0 {
            if started {
                zero = true;
            }
            continue;
        }
        if zero {
            out.push_str("零");
            zero = false;
        }
        // inner 一十 → 十? No: 一百一十 keeps 一十. Only strip at very start
        // (handled by caller). "十" alone inside group: 10 → 一十 handled
        // by caller strip; 1010 → 一千零一十? Standard: 一千零一十.
        // Keep 一十 inside groups (一千零一十, not 一千零十 — both heard,
        // former is standard). Simplify: tens digit 1 → 十 without 一?
        // Common speech: 一百一十, 一千零一十. So keep 一.
        out.push_str(D[x]);
        out.push_str(u[i]);
        started = true;
    }
    // group "10" → 一十 (caller strips leading 一十 only at string start)
    out
}

fn digitwise(s: &str) -> String {
    s.chars().map(|c| D[c.to_digit(10).unwrap() as usize]).collect()
}

/// Normalize one text. Idempotent-ish for already-Chinese prose.
pub fn normalize(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        // range minus / leading minus
        if (c == '-' || c == '－') && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
            let prev_digit = i > 0 && (chars[i - 1].is_ascii_digit() || chars[i - 1] == '.');
            if prev_digit {
                // N-M handled by the number parser's range lookahead; a lone
                // '-' between digits here means parser already passed — emit 到
                out.push_str("到");
            } else {
                out.push_str("负");
            }
            i += 1;
            continue;
        }
        if c.is_ascii_digit() {
            let (word, j) = parse_number(&chars, i);
            out.push_str(&word);
            i = j;
            continue;
        }
        match c {
            // lone % (not after digits — digit trails are consumed by the
            // parser): keep position, best-effort.
            '％' | '%' => out.push_str("百分之"),
            '°' => out.push_str("度"),
            '×' | '*' => out.push_str("乘"),
            '÷' => out.push_str("除以"),
            '~' | '～' => out.push_str("到"),
            _ => out.push(c),
        }
        i += 1;
    }
    out
}

/// Parse a number starting at chars[i] (chars[i] is a digit).
/// Returns (reading, next index). Consumes optional .frac, % suffix,
/// and N~M / N-M ranges (emits 到 + second number, recursive once).
fn parse_number(chars: &[char], i: usize) -> (String, usize) {
    let mut j = i;
    while j < chars.len() && chars[j].is_ascii_digit() {
        j += 1;
    }
    let int_part = chars[i..j].iter().collect::<String>();
    // decimal fraction?
    let mut frac: Option<String> = None;
    if j < chars.len() && chars[j] == '.' && j + 1 < chars.len() && chars[j + 1].is_ascii_digit() {
        let mut k = j + 1;
        while k < chars.len() && chars[k].is_ascii_digit() {
            k += 1;
        }
        frac = Some(chars[j + 1..k].iter().collect());
        j = k;
    }
    // percent suffix (space-tolerant: "35 %" → 百分之三十五)?
    let mut k = j;
    while k < chars.len() && (chars[k] == ' ' || chars[k] == '　') {
        k += 1;
    }
    let mut pct = false;
    if k < chars.len() && (chars[k] == '%' || chars[k] == '％') {
        pct = true;
        j = k + 1;
    }
    // year: exactly 4 digits + 年 → digit-by-digit
    let is_year = int_part.len() == 4 && frac.is_none() && j < chars.len() && chars[j] == '年';
    let mut word = if is_year {
        digitwise(&int_part)
    } else {
        int_part.parse::<u64>().map(int_zh).unwrap_or_else(|_| digitwise(&int_part))
    };
    if let Some(f) = frac {
        word.push_str("点");
        word.push_str(&digitwise(&f));
    }
    if pct {
        word = format!("百分之{word}");
    }
    // range: sep + digits → 到 + number
    if j < chars.len() && matches!(chars[j], '~' | '～' | '-' | '－' | '–')
        && j + 1 < chars.len()
        && chars[j + 1].is_ascii_digit()
    {
        let (w2, j2) = parse_number(chars, j + 1);
        word.push_str("到");
        word.push_str(&w2);
        j = j2;
    }
    (word, j)
}
