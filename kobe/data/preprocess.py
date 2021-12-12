import hashlib
import json
import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import sentencepiece as spm
import webdataset as wds
from joblib import Parallel, delayed
from tqdm import tqdm

from kobe.data.dataset import Example


@dataclass
class RawExample:
    title: str
    description: str
    condition: str
    fact: str


FIELDS = ["title", "cond", "desc", "fact"]


def add_options(parser: ArgumentParser):
    # fmt: off
    parser.add_argument("--raw-path", default="data-v2/raw/")
    parser.add_argument("--processed-path", default="data-v2/processed/")
    parser.add_argument("--split", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--vocab-file", default="data-v2/vocab.text.model")
    parser.add_argument("--cond-vocab-file", default="data-v2/vocab.cond.model")
    # fmt: on


def prepare_file(args: Namespace):
    for field in FIELDS:
        for split in args.split:
            assert os.path.exists(os.path.join(args.raw_path, f"{split}.{field}"))
    os.makedirs(os.path.dirname(args.processed_path), exist_ok=True)


def preprocess_raw_example(
    rawe: RawExample,
    tokenizer: spm.SentencePieceProcessor,
    cond_tokenizer: spm.SentencePieceProcessor,
) -> Tuple[str, Example]:

    e = Example(
        title_token_ids=tokenizer.EncodeAsIds(rawe.title),
        description_token_ids=tokenizer.EncodeAsIds(rawe.description),
        condition_token_ids=cond_tokenizer.EncodeAsIds(rawe.condition),
        fact_token_ids=tokenizer.EncodeAsIds(rawe.fact),
        description=rawe.description,
    )
    return hashlib.sha1(json.dumps(e.__dict__).encode()).hexdigest(), e


def preprocess_raw(
    input_prefix: str,
    output: str,
    text_tokenizer: spm.SentencePieceProcessor,
    cond_tokenizer: spm.SentencePieceProcessor,
):
    contents = {field: open(f"{input_prefix}.{field}").readlines() for field in FIELDS}
    # ensure the fields have the same number of rows
    N = len(contents[FIELDS[0]])
    assert all([len(contents[field]) == N for field in FIELDS])
    raw_examples = [
        RawExample(
            title=contents["title"][idx],
            description=contents["desc"][idx],
            condition=contents["cond"][idx],
            fact=contents["fact"][idx],
        )
        for idx in range(N)
    ]
    # tokenize the texts in parallel
    examples = Parallel(n_jobs=8)(
        delayed(preprocess_raw_example)(rawe, text_tokenizer, cond_tokenizer)
        for rawe in tqdm(raw_examples)
    )
    np.random.shuffle(examples)
    # store the processed samples
    with wds.TarWriter(output) as dst:
        for key, e in tqdm(examples):
            sample = {"__key__": key, "json": e.__dict__}
            dst.write(sample)


if __name__ == "__main__":
    parser = ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    prepare_file(args)

    text_tokenizer = spm.SentencePieceProcessor()
    text_tokenizer.Load(args.vocab_file)
    cond_tokenizer = spm.SentencePieceProcessor()
    cond_tokenizer.Load(args.cond_vocab_file)

    for split in args.split:
        preprocess_raw(
            input_prefix=os.path.join(args.raw_path, split),
            output=os.path.join(args.processed_path, f"{split}.tar"),
            text_tokenizer=text_tokenizer,
            cond_tokenizer=cond_tokenizer,
        )
