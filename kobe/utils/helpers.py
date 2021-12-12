import subprocess
from tempfile import NamedTemporaryFile
from typing import List

import sentencepiece as spm

BASELINE = "baseline"
KOBE_ATTRIBUTE = "kobe-attr"
KOBE_KNOWLEDGE = "kobe-know"
KOBE_FULL = "kobe-full"


def get_vocab_size(vocab_path: str) -> int:
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(vocab_path)
    return len(tokenizer)


def bleu(reference: List[str], candidate: List[str]) -> float:
    """
    return the bleu score, including multi-bleu and nist bleu.
    :param reference: reference
    :param candidate: candidate
    :return: bleu score
    """
    ref_f = NamedTemporaryFile("w")
    ref_f.write("\n".join(reference))
    cand_f = NamedTemporaryFile("w")
    cand_f.write("\n".join(candidate))
    result_f = NamedTemporaryFile("r")

    command = (
        "perl scripts/multi-bleu.perl -lc "
        + ref_f.name
        + "<"
        + cand_f.name
        + "> "
        + result_f.name
    )
    subprocess.call(command, shell=True)
    result = result_f.read()

    print(result)

    return float(result.split()[2][:-1])
