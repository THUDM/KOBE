import os
import pickle as pkl
import numpy as np
import argparse

vocab = {'<pad>': 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--item_list_path', type=str, default='data/graph/item.txt')
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--instance_path', type=str, default='data/emotion/')
    parser.add_argument('--output_path', type=str, default='data/emotion/preprocessed/knowledge.pkl')
    args = parser.parse_args()

    item_to_node_list = {}
    with open(args.item_list_path) as f:
        for line in f.readlines():
            item_id_hash, title = line.split('\t')
            keywords = title.split()
            item_to_node_list[item_id_hash] = keywords
            for keyword in keywords:
                if keyword not in vocab:
                    vocab[keyword] = len(vocab)
    
    print(len(vocab))

    # dummy_knowledge = dict([(word, np.zeros(args.hidden_size)) for word in vocab])
    # pkl.dump(dummy_knowledge, open(args.output_path, 'wb'))

    splits = ['valid', 'test', 'train']

    for split in splits:
        with open(os.path.join(args.instance_path, split + '.hash_id')) as f:
            instances_hashids = f.read().strip().split('\n')
        with open(os.path.join(args.instance_path, split + '.src')) as f:
            instance_srcs = f.read().strip().split('\n')
        with open(os.path.join(args.instance_path, split + '.tgt')) as f:
            instance_tgts = f.read().strip().split('\n')
        print(len(instances_hashids))
        with open(os.path.join(args.instance_path, split + '.matched_knowledge'), 'w') as f:
            for src, tgt, hashid in zip(instance_srcs, instance_tgts, instances_hashids):
                if src != "" and tgt != "":
                    f.write(' '.join([str(vocab[node]) for node in item_to_node_list[hashid]]) + '\n')
