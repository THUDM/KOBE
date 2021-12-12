import sentencepiece as spm
import torch

BASELINE = "baseline"
KOBE_ATTRIBUTE = "kobe-attr"
KOBE_KNOWLEDGE = "kobe-know"
KOBE_FULL = "kobe-full"


def get_vocab_size(vocab_path: str) -> int:
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(vocab_path)
    return len(tokenizer)


# Metrics
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    assert logits.dim() == 2
    assert targets.dim() == 1
    pred = logits.argmax(dim=1)
    return (pred == targets).sum().item() / targets.shape[0]
