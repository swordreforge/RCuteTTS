//! tts prompt tokenizer: SentencePiece + special-token splitting.
//!
//! Mirrors `CuteTTSSentencePieceTokenizer` + HF `encode()` for the tts path:
//! HF splits on special tokens FIRST (`<|im_start|>`→4, `<|im_end|>`→5,
//! `<|endofprompt|>`→16384), SentencePiece-encodes each gap as ONE string
//! (segmentation is context-sensitive — encoding the whole string at once
//! gives different boundary pieces, caught by the tricky fixture).
//! `<|im_start|>`/`<|im_end|>` are also native SP pieces, but HF's
//! special-split wins when the literal appears in text.
//! No BOS is added (`build_inputs_with_special_tokens` is identity).
//!
//! Prompt: "Transform the text into speech output.\ntext input:\n{text}\n<|endofprompt|>"

use sentencepiece::SentencePieceProcessor;

pub const ENDOFPROMPT_STR: &str = "<|endofprompt|>";
pub const ENDOFPROMPT_ID: i64 = 16384;

/// Special literals handled before SP (longest match first).
const SPECIALS: [(&str, i64); 3] = [
    ("<|endofprompt|>", 16384),
    ("<|im_start|>", 4),
    ("<|im_end|>", 5),
];

pub struct PromptTokenizer {
    sp: SentencePieceProcessor,
}

impl PromptTokenizer {
    pub fn load(model_path: &std::path::Path) -> Self {
        let sp = SentencePieceProcessor::open(model_path)
            .unwrap_or_else(|e| panic!("load sentencepiece {}: {e}", model_path.display()));
        PromptTokenizer { sp }
    }

    pub fn piece_size(&self) -> usize {
        self.sp.len()
    }

    /// Encode one segment (no added tokens inside) to ids.
    pub fn encode_segment(&self, text: &str) -> Vec<i64> {
        self.sp.encode(text).unwrap_or_else(|e| panic!("sp encode failed: {e}")).into_iter().map(|p| p.id as i64).collect()
    }

    /// Full HF-compatible encode: split on special literals first.
    pub fn encode(&self, text: &str) -> Vec<i64> {
        let mut ids = Vec::new();
        let mut rest = text;
        'outer: while !rest.is_empty() {
            let mut best: Option<(usize, usize, i64)> = None; // (pos, len, id)
            for (lit, id) in SPECIALS {
                if let Some(pos) = rest.find(lit) {
                    if best.map(|b| pos < b.0).unwrap_or(true) {
                        best = Some((pos, lit.len(), id));
                    }
                }
            }
            match best {
                Some((pos, len, id)) => {
                    ids.extend(self.encode_segment(&rest[..pos]));
                    ids.push(id);
                    rest = &rest[pos + len..];
                    continue 'outer;
                }
                None => break,
            }
        }
        ids.extend(self.encode_segment(rest));
        ids
    }

    /// tts prompt assembly (processor._text_only_prompt).
    pub fn tts_prompt(text: &str) -> String {
        format!("Transform the text into speech output.\ntext input:\n{text}\n<|endofprompt|>")
    }

    pub fn encode_tts(&self, text: &str) -> Vec<i64> {
        self.encode(&Self::tts_prompt(text))
    }
}
