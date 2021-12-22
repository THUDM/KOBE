import sentencepiece as spm
import torch
import torch.nn.functional as F
from transformers.models.bert.tokenization_bert import BertTokenizer

BASELINE = "baseline"
KOBE_ATTRIBUTE = "kobe-attr"
KOBE_KNOWLEDGE = "kobe-know"
KOBE_FULL = "kobe-full"


def get_bert_vocab_size(vocab_path: str) -> int:
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    return tokenizer.vocab_size


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


def top_k_top_p_sampling(
    logits, top_k=0, top_p=0.0, temperature=1, filter_value=-float("Inf")
) -> int:
    """Sample from a filtered distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    logits /= temperature
    assert (
        logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    # Sample from the filtered distribution
    probabilities = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probabilities, 1)

    return int(next_token.item())


def diversity(tokenized_lines, n=4) -> int:
    """Defined as the unique number of ngrams generated on the test set."""
    n_grams_all = []
    for line in tokenized_lines:
        n_grams = list(zip(*[line[i:] for i in range(n)]))
        n_grams_all += n_grams

    return len(set(n_grams_all))
