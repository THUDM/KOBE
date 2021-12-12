from argparse import ArgumentParser, Namespace

from kobe.utils import helpers


def add_options(parser: ArgumentParser):
    # fmt: off
    # Dataset
    parser.add_argument("--train-data", default="data-v2/processed/train-*.tar", type=str)
    parser.add_argument("--valid-data", default="data-v2/processed/valid-*.tar", type=str)
    parser.add_argument("--test-data", default="data-v2/processed/test-*.tar", type=str)
    parser.add_argument("--text-vocab-path", default="data-v2/vocab.text.model", type=str)
    parser.add_argument("--cond-vocab-path", default="data-v2/vocab.cond.model", type=str)
    parser.add_argument("--num-workers", default=8, help="Number of data loaders", type=int)

    # Model
    parser.add_argument("--d-model", default=512, type=int)
    parser.add_argument("--nhead", default=8, type=int)
    parser.add_argument("--num-encoder-layers", default=6, type=int)
    parser.add_argument("--num-decoder-layers", default=6, type=int)
    parser.add_argument("--max-seq-len", default=512, type=int)
    parser.add_argument("--mode", default="baseline", type=str, choices=[
        helpers.BASELINE, helpers.KOBE_ATTRIBUTE, helpers.KOBE_KNOWLEDGE, helpers.KOBE_FULL])

    # Training
    parser.add_argument("--name", default="exp", type=str, help="expeirment name")
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument("--grad-clip", default=1.0, type=float, help="clip threshold of gradients")
    parser.add_argument("--epochs", default=300, type=int, help="number of epochs to train")
    parser.add_argument("--patience", default=100, type=int, help="early stopping patience")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout rate")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--seed", default=42, type=int)

    # Evaluation
    parser.add_argument("--test", action="store_true", help="only do evaluation")
    parser.add_argument("--load-file", required=False, type=str)
    parser.add_argument("--beam-size", default=0, type=int)

    # fmt: on


def add_args(args: Namespace):
    args.text_vocab_size = helpers.get_vocab_size(args.text_vocab_path)
    args.cond_vocab_size = helpers.get_vocab_size(args.cond_vocab_path)
