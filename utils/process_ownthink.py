import os
import argparse
import pickle as pkl
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
from utils.clean import get_chinese

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/graph/ownthink.pkl')
    parser.add_argument('--item_list_path', type=str,
                        default='data/graph/item.txt')
    parser.add_argument('--instance_path', type=str, default='data/emotion/')
    args = parser.parse_args()

    vocab = {'<pad>': 0}
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

    ownthink = pkl.load(open(args.input, 'rb'))
    ownthink = dict([(k, ownthink[k])
                     for k in ownthink
                     if ownthink[k]['message'] != 'error' and '请求异常' not in ownthink[k]['data']])
    abstract = dict([(k, get_chinese(ownthink[k]['data']['desc'])[:50])
                     for k in ownthink
                     if len(get_chinese(ownthink[k]['data']['desc'])) > 10])
    print(len(abstract))

    # map supporting fact chars to target vocabulary
    fact_dict = {}
    with open(os.path.join(args.instance_path, 'preprocessed/tgt.dict')) as f:
        for i, word in enumerate(f.read().strip().split('\n')):
            fact_dict[word.split()[0]] = i
    abstract_id = defaultdict(list)
    for k in abstract:
        for x in list(abstract[k]):
            abstract_id[k].append(
                fact_dict[x] if x in fact_dict else fact_dict['<unk>'])

    # Calculating document frequency for sampling
    df = defaultdict(int)
    N = len(item_to_node_list)
    for itemid in item_to_node_list:
        node_list = [node for node in item_to_node_list[itemid]
                     if node in abstract]
        for node in set(node_list):
            df[node] += 1

    def sample_supporting_fact(node_list):
        node_list = [node for node in node_list if node in abstract]
        tfidf = []
        nodes = []
        for node in set(node_list):
            # Discard rare words
            if df[node] > 5 and len(node) > 1:
                tfidf.append(node_list.count(node) * np.log(N / df[node]))
                nodes.append(node)
        # print(np.random.choice(nodes, size=2, replace=False, p=tfidf / np.sum(tfidf)), nodes)
        if nodes:
            if len(nodes) > 4:
                sampled = sorted(np.random.choice(len(nodes), 4, replace=False, p=tfidf / np.sum(tfidf)))
                sampled = [nodes[idx] for idx in sampled]
            else:
                sampled = nodes
        else:
            # Sample one word from the item title and use its supporting fact
            sampled = [np.random.choice(list(abstract.keys()))]
        return sampled

    # for itemid in item_to_node_list:
    #     node_list = item_to_node_list[itemid]

    splits = ['valid', 'test', 'train']
    for split in splits:
        with open(os.path.join(args.instance_path, split + '.hash_id')) as f:
            instances_hashids = f.read().strip().split('\n')
        with open(os.path.join(args.instance_path, split + '.src')) as f:
            instance_srcs = f.read().strip().split('\n')
        with open(os.path.join(args.instance_path, split + '.tgt')) as f:
            instance_tgts = f.read().strip().split('\n')
        with open(os.path.join(args.instance_path, split + '.supporting_facts'), 'w') as f, open(os.path.join(args.instance_path, split + '.supporting_facts_str'), 'w') as f_str:
            for src, tgt, hashid in tqdm(zip(instance_srcs, instance_tgts, instances_hashids)):
                if src != "" and tgt != "":
                    sampled_list = sample_supporting_fact(item_to_node_list[hashid]) 
                    for sampled in sampled_list:
                        f.write(' '.join(map(str, abstract_id[sampled])) + ' ')
                        f_str.write(' '.join(abstract[sampled]) + ' ')
                    f.write('\n')
                    f_str.write('\n')
