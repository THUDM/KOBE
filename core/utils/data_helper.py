import pickle as pkl
import linecache
from random import Random

import numpy as np
import torch
import torch.utils.data as torch_data

import utils

num_samples = 1


class BiTestDataset(torch_data.Dataset):

    def __init__(self, infos, indices=None):

        self.src_id = infos['src_id']
        self.src_str = infos['src_str']
        self.tgt_id = infos['tgt_id']
        self.tgt_str = infos['tgt_str']
        self.length = infos['length']
        self.infos = infos
        if indices is None:
            self.indices = list(range(self.length))
        else:
            self.indices = indices

    def __getitem__(self, index):
        index = self.indices[index]
        src = self.src_id[index]
        original_src = self.src_str[index]
        tgt = self.tgt_id[index]
        original_tgt = self.tgt_str[index]

        return src, tgt, original_src, original_tgt

    def __len__(self):
        return len(self.indices)


class BiDataset(torch_data.Dataset):

    def __init__(self, infos, indices=None, char=False):

        self.srcF = infos['srcF']
        self.tgtF = infos['tgtF']
        self.original_srcF = infos['original_srcF']
        self.original_tgtF = infos['original_tgtF']
        self.length = infos['length']
        self.infos = infos
        self.char = char
        if indices is None:
            self.indices = list(range(self.length))
        else:
            self.indices = indices

    def __getitem__(self, index):
        index = self.indices[index]
        src = list(map(int, linecache.getline(
            self.srcF, index+1).strip().split()))
        tgt = list(map(int, linecache.getline(
            self.tgtF, index+1).strip().split()))
        original_src = linecache.getline(
            self.original_srcF, index+1).strip().split()
        original_tgt = linecache.getline(self.original_tgtF, index+1).strip().split() if not self.char else \
            list(linecache.getline(self.original_tgtF, index + 1).strip())

        return src, tgt, original_src, original_tgt

    def __len__(self):
        return len(self.indices)


class BiKnowledgeDataset(BiDataset):
    """Knowledge is a tensor with shape [knowledge_len, hidden size]"""

    def __init__(self, matched_knowledge_path, **kwargs):
        BiDataset.__init__(self, **kwargs)
        # self.knowledge = pkl.load(open(knowledge_path, 'rb'))

        self.matched_knowledge_path = matched_knowledge_path
        # with open(matched_knowledge_path) as f:
        #     self.matched_knowledge_list = [line.split() for line in f.read().strip().split('\n')]
        # assert len(self.matched_knowledge_list) == self.length

    def __getitem__(self, index):
        src, tgt, original_src, original_tgt = BiDataset.__getitem__(self, index)
        knowledge = list(map(int, linecache.getline(self.matched_knowledge_path, index+1).strip().split()))
        # knowledge = np.stack([self.knowledge[node]
        #                       for node in self.matched_knowledge_list[index]
        #                       if node in self.knowledge])
        return src, tgt, original_src, original_tgt, knowledge


def splitDataset(data_set, sizes):
    length = len(data_set)
    indices = list(range(length))
    rng = Random()
    rng.seed(1234)
    rng.shuffle(indices)

    data_sets = []
    part_len = int(length / sizes)
    for i in range(sizes-1):
        data_sets.append(BiDataset(data_set.infos, indices[0:part_len]))
        indices = indices[part_len:]
    data_sets.append(BiDataset(data_set.infos, indices))
    return data_sets


def padding(data):
    src, tgt, original_src, original_tgt, knowledge = zip(*data)

    src_len = [len(s) for s in src]
    knowledge_len = [len(s) for s in knowledge]
    # char_word_pad = max(max(src_len), max(knowledge_len))
    src_pad = torch.zeros(len(src), max(src_len)).long()
    for i, s in enumerate(src):
        end = src_len[i]
        src_pad[i, :end] = torch.LongTensor(s[end-1::-1])

    tgt_len = [len(s) for s in tgt]
    tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
    for i, s in enumerate(tgt):
        end = tgt_len[i]
        tgt_pad[i, :end] = torch.LongTensor(s)[:end]

    knowledge_pad = torch.zeros(len(knowledge), max(knowledge_len)).long()
    for i, s in enumerate(knowledge):
        end = knowledge_len[i]
        knowledge_pad[i, :end] = torch.LongTensor(s[end-1::-1])

    return src_pad, tgt_pad, \
           torch.LongTensor(src_len), torch.LongTensor(tgt_len), \
           original_src, original_tgt, knowledge_pad, torch.LongTensor(knowledge_len)


def ae_padding(data):
    src, tgt, original_src, original_tgt = zip(*data)

    src_len = [len(s) for s in src]
    src_pad = torch.zeros(len(src), max(src_len)).long()
    for i, s in enumerate(src):
        end = src_len[i]
        src_pad[i, :end] = torch.LongTensor(s)[:end]

    tgt_len = [len(s) for s in tgt]
    tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
    for i, s in enumerate(tgt):
        end = tgt_len[i]
        tgt_pad[i, :end] = torch.LongTensor(s)[:end]

    ae_len = [len(s)+2 for s in src]
    ae_pad = torch.zeros(len(src), max(ae_len)).long()
    for i, s in enumerate(src):
        end = ae_len[i]
        ae_pad[i, 0] = utils.BOS
        ae_pad[i, 1:end-1] = torch.LongTensor(s)[:end-2]
        ae_pad[i, end-1] = utils.EOS

    return src_pad, tgt_pad, ae_pad, \
        torch.LongTensor(src_len), torch.LongTensor(tgt_len), torch.LongTensor(ae_len), \
        original_src, original_tgt


def split_padding(data):
    src, tgt, original_src, original_tgt = zip(*data)

    split_samples = []
    num_per_sample = int(len(src) / utils.num_samples)

    for i in range(utils.num_samples):
        split_src = src[i*num_per_sample:(i+1)*num_per_sample]
        split_tgt = tgt[i*num_per_sample:(i+1)*num_per_sample]
        split_original_src = original_src[i *
                                          num_per_sample:(i + 1) * num_per_sample]
        split_original_tgt = original_tgt[i *
                                          num_per_sample:(i + 1) * num_per_sample]

        src_len = [len(s) for s in split_src]
        src_pad = torch.zeros(len(split_src), max(src_len)).long()
        for i, s in enumerate(split_src):
            end = src_len[i]
            src_pad[i, :end] = torch.LongTensor(s)[:end]

        tgt_len = [len(s) for s in split_tgt]
        tgt_pad = torch.zeros(len(split_tgt), max(tgt_len)).long()
        for i, s in enumerate(split_tgt):
            end = tgt_len[i]
            tgt_pad[i, :end] = torch.LongTensor(s)[:end]

        split_samples.append([src_pad, tgt_pad,
                              torch.LongTensor(
                                  src_len), torch.LongTensor(tgt_len),
                              split_original_src, split_original_tgt])

    return split_samples
