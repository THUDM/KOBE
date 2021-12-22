import tempfile
from argparse import ArgumentParser

import sentencepiece as spm
from transformers.models.bert.tokenization_bert import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

BOS_TOKEN = tokenizer.cls_token
EOS_TOKEN = tokenizer.sep_token
UNK_TOKEN = tokenizer.unk_token
PAD_ID = tokenizer.pad_token_id
BOS_ID = tokenizer.cls_token_id
EOS_ID = tokenizer.sep_token_id
UNK_ID = tokenizer.unk_token_id

if __name__ == "__main__":
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--vocab-file", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, default=5000)
    parser.add_argument("--algo", type=str, default="bpe", choices=["bpe", "word"])
    # fmt: on
    args = parser.parse_args()
    print("Building token vocabulary")
    with tempfile.NamedTemporaryFile("w") as f:
        # concatenate input files
        for input_fname in args.input:
            with open(input_fname) as input_f:
                f.write(input_f.read() + "\n")
        # run sentence piece with bpe
        spm.SentencePieceTrainer.Train(
            f"--add_dummy_prefix=false --pad_id={PAD_ID} --bos_id={BOS_ID} --eos_id={EOS_ID} --unk_id={UNK_ID} "
            f"--vocab_size={args.vocab_size} "
            f"--model_prefix={args.vocab_file} --model_type={args.algo} "
            f"--input={f.name}"
        )
