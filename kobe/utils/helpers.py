import sentencepiece as spm

BASELINE = "baseline"
KOBE_ATTRIBUTE = "kobe-attr"
KOBE_KNOWLEDGE = "kobe-know"
KOBE_FULL = "kobe-full"


def get_vocab_size(vocab_path: str) -> int:
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(vocab_path)
    return len(tokenizer)
