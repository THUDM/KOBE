#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-15 15:41:14
# @Author  : Junyang Lin
# @Version : $0$

import numpy as np
import random
import codecs
import utils
import models
import opts
from collections import OrderedDict, defaultdict
import time
import pickle
import argparse
import sys
import os
import json
import lr_scheduler as L
from torch.nn.init import xavier_uniform_
import torch.utils.data
import torch

import matplotlib
matplotlib.use('Agg')
from matplotlib import ticker
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='train.py')
opts.model_opts(parser)

opt = parser.parse_args()
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)
random.seed(opt.seed)
np.random.seed(opt.seed)

opts.convert_to_config(opt, config)

if opt.label_dict_file != '':
    with open(opt.label_dict_file, 'r') as f:
        label_dict = json.load(f)


def build_log():
    # log
    if not os.path.exists(config.logF):
        os.mkdir(config.logF)
    if opt.log == '':
        log_path = config.logF + str(int(time.time() * 1000)) + '/'
    else:
        log_path = config.logF + opt.log + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    print_log = utils.print_log(log_path + 'log.txt')
    return print_log, log_path


# build_log()

# cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
config.use_cuda = use_cuda
if use_cuda:
    torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True


def makeTestData(srcFile, tgtFile, srcDicts, tgtDicts, opt=None, trun=True):
    sizes = 0
    count, empty_ignored, limit_ignored = 0, 0, 0
    opt = opt

    srcF = codecs.open(srcFile, encoding='utf-8')
    tgtF = codecs.open(tgtFile, encoding='utf-8')
    srcIdL, srcStrL, = [], []
    tgtIdL, tgtStrL, = [], []

    if opt.label_dict_file:
        with open(opt.label_dict_file, 'r') as f:
            LabelDict = json.load(f)

    if opt.rm:
        rm = opt.rm

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            print('WARNING: source and target do not have the same number of sentences')
            break

        sline = sline.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            empty_ignored += 1
            continue

        if opt.lower:
            sline = sline.lower()
            tline = tline.lower()

        srcWords = sline.split()
        tgtWords = tline.split()


        if (opt.src_filter == 0 or len(sline.split()) <= opt.src_filter) and \
           (opt.tgt_filter == 0 or len(tline.split()) <= opt.tgt_filter):
            if opt.src_trun > 0 and trun:
                srcWords = srcWords[:opt.src_trun]
            if opt.tgt_trun > 0 and trun:
                tgtWords = tgtWords[:opt.tgt_trun]

            srcIds = srcDicts.convertToIdx(srcWords, utils.UNK_WORD)
            tgtIds = tgtDicts.convertToIdx(
                tgtWords, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD)

            srcIdL.append(srcIds)
            tgtIdL.append(tgtIds)
            srcStrL.append(srcWords)
            tgtStrL.append(tgtWords)

            sizes += 1
        else:
            limit_ignored += 1

        count += 1

    srcF.close()
    tgtF.close()

    print('Prepared %d sentences (%d and %d ignored due to length == 0 or > )' %
          (sizes, empty_ignored, limit_ignored))

    test = {'src_str': srcStrL, 'src_id': srcIdL, 
            'tgt_str': tgtStrL, 'tgt_id': tgtIdL,
            'length': sizes}
    testset = utils.BiTestDataset(test)

    return testset


def load_data(test_src, test_tgt, opt=None):
    print('loading data...\n')
    data = pickle.load(open(config.data + 'data.pkl', 'rb'))
    src_vocab = data['dict']['src']
    config.src_vocab_size = src_vocab.size()
    tgt_vocab = data['dict']['tgt']
    config.tgt_vocab_size = tgt_vocab.size()

    testset = makeTestData(test_src, test_tgt, src_vocab, tgt_vocab, opt=opt)

    if hasattr(config, 'test_batch_size'):
        test_batch_size = config.test_batch_size
    else:
        test_batch_size = config.batch_size

    testloader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=test_batch_size,
                                             shuffle=True,
                                             num_workers=0,
                                             collate_fn=utils.padding)

    return {'testset': testset, 'testloader': testloader, 
            'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab}


def build_model(checkpoints, print_log):
    # for k, v in config.items():
    #     print_log("%s:\t%s\n" % (str(k), str(v)))

    # model
    print('building model...\n')
    model = getattr(models, config.model)(config,
                                          src_padding_idx=utils.PAD,
                                          tgt_padding_idx=utils.PAD,
                                          label_smoothing=config.label_smoothing,
                                          tgt_vocab=None)
    if config.param_init != 0.0:
        for p in model.parameters():
            p.data.uniform_(-config.param_init, config.param_init)
    if config.param_init_glorot:
        for p in model.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
    if checkpoints is not None:
        model.load_state_dict(checkpoints['model'])
    if opt.pretrain:
        print('loading checkpoint from %s' % opt.pretrain)
        pre_ckpt = torch.load(
            opt.pretrain, map_location=lambda storage, loc: storage)['model']
        model.load_state_dict(pre_ckpt)
        # pre_ckpt = OrderedDict({key[8:]: pre_ckpt[key]
        #                         for key in pre_ckpt if key.startswith('encoder')})
        # print(model.encoder.state_dict().keys())
        # print(pre_ckpt.keys())
        # model.encoder.load_state_dict(pre_ckpt)
    if use_cuda:
        model.cuda()

    # optimizer
    if checkpoints is not None:
        optim = checkpoints['optim']
    else:
        optim = models.Optim(config.optim,
                             config.learning_rate, config.max_grad_norm,
                             lr_decay=config.learning_rate_decay,
                             start_decay_steps=config.start_decay_steps,
                             beta1=config.beta1, beta2=config.beta2,
                             decay_method=config.decay_method,
                             warmup_steps=config.warmup_steps,
                             model_size=config.hidden_size)
    optim.set_parameters(model.parameters())

    # print log
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    # for k, v in config.items():
    #     print_log("%s:\t%s\n" % (str(k), str(v)))
    # print_log("\n")
    # print_log(repr(model) + "\n\n")
    # print_log('total number of parameters: %d\n\n' % param_count)

    return model, optim, print_log

def eval_model(model, data, params):

    model.eval()
    reference, candidate, source, alignments = [], [], [], []
    weights = []
    count, total_count = 0, len(data['testset'])
    testloader = data['testloader']
    tgt_vocab = data['tgt_vocab']

    for src, tgt, src_len, tgt_len, original_src, original_tgt in testloader:

        if config.use_cuda:
            src = src.cuda()
            src_len = src_len.cuda()

        with torch.no_grad():
            if config.beam_size > 1:
                samples, alignment, weight = model.beam_sample(
                    src, src_len, beam_size=config.beam_size, eval_=True)
            else:
                samples, alignment, weight = model.sample(src, src_len)

        candidate += [tgt_vocab.convertToLabels(s, utils.EOS) for s in samples]
        source += original_src
        reference += original_tgt
        if alignment is not None:
            alignments += [align for align in alignment]
        if weight is not None:
            weights += [w.cpu() for w in weight]

        count += len(original_src)
        utils.progress_bar(count, total_count)

    i = 0
    for s, c, weight in zip(source[50:100], candidate[50:100], weights[50:100]):
        showAttention(params['log_path'], s[60:90], c, weight[:len(c), 60:90], i)
        i += 1

    if config.unk and config.attention != 'None':
        cands = []
        for s, c, align in zip(source, candidate, alignments):
            cand = []
            for word, idx in zip(c, align):
                if word == utils.UNK_WORD and idx < len(s):
                    try:
                        cand.append(s[idx])
                    except:
                        cand.append(word)
                        print("%d %d\n" % (len(s), idx))
                else:
                    cand.append(word)
            cands.append(cand)
            if len(cand) == 0:
                print('Error!')
        candidate = cands

    with codecs.open(params['log_path'] + 'test_candidate.txt', 'w+', 'utf-8') as f:
        for i in range(len(candidate)):
            f.write(" ".join(candidate[i]) + '\n')
    if opt.label_dict_file != '':
        results = utils.eval_metrics(
            reference, candidate, label_dict, params['log_path'])
    score = {}
    result_line = ""
    for metric in config.metrics:
        if opt.label_dict_file != '':
            score[metric] = results[metric]
            result_line += metric + ": %s " % str(score[metric])
        else:
            score[metric] = getattr(utils, metric)(
                reference, candidate, params['log_path'], params['log'], config)

    if opt.label_dict_file != '':
        result_line += '\n'
        params['log'](result_line)

    return score


def showAttention(path, s, c, attentions, index):
    plt.tick_params(labelsize=20)
    # Set up figure with colorbar
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    figsize = 20, 10
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)
    # Set up axes
    ax.set_xticklabels([''] + s, rotation=45)
    ax.set_yticklabels([''] + c)
    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # plt.show()
    plt.savefig(path + str(index) + '.jpg')


def main():
    # checkpoint
    if opt.restore:
        print('loading checkpoint...\n')
        checkpoints = torch.load(
            opt.restore, map_location=lambda storage, loc: storage)
    else:
        checkpoints = None
    
    test_src, test_tgt = config.test_src, config.test_tgt
    data = load_data(test_src, test_tgt, opt=config)

    print_log, log_path = build_log()
    model, optim, print_log = build_model(checkpoints, print_log)
    params = {'log': print_log, 'log_path': log_path}

    score = eval_model(model, data, params)


if __name__ == '__main__':
    main()

