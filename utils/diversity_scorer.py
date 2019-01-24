import argparse
import numpy as np
from tqdm import tqdm


def generate_n_grams(x, n):
    n_grams = set(zip(*[x[i:] for i in range(n)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x


def distinct_n_grams(tokenized_lines, n):

    n_grams_all = []
    for line in tokenized_lines:
        n_grams = generate_n_grams(line, n)
        n_grams_all += n_grams

    return len(set(n_grams_all)), len(set(n_grams_all)) / len(n_grams_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    args = parser.parse_args()
    with open(args.input) as f:
        lines = f.read().strip().split('\n')
        tokenized = [line.split() for line in lines]

    for n in range(1, 6):
        cnt, percent = distinct_n_grams(tokenized, n)
        print(f'Distinct {n}-grams (cnt, percentage) = ({cnt}, {percent:.3f})')
