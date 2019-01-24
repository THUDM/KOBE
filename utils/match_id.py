import os
import argparse
import pickle as pkl
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
from utils.clean import get_chinese

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='data/emotion/all.src')
    parser.add_argument('--id', type=str, default='data/emotion/all.hash_id')
    # parser.add_argument('--input', type=str, default='data/finals/baseline/train.src')
    inputs = ['data/finals/baseline/valid.src',
              'data/finals/baseline/train.src',
              'data/finals/baseline/test.src']
    args = parser.parse_args()

    with open(args.src) as f:
        srcs = [line[4:] for line in f.read().strip().split('\n')]
    with open(args.id) as f:
        hash_ids = f.read().strip().split('\n')
    
    id_dict = {}
    for src, hash_id in zip(srcs, hash_ids):
        if src not in id_dict:
            id_dict[src] = hash_id
    
    for input in inputs:
        with open(input) as f:
            input_srcs = f.read().strip().split('\n')
        with open(input[:-3] + 'hash_id', 'w') as fout:
            for input_src in input_srcs:
                fout.write(id_dict[input_src] + '\n')
