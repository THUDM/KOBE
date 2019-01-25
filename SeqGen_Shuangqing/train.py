import torch
import torch.utils.data
from torch.nn.init import xavier_uniform_
import lr_scheduler as L

import os
import sys
import argparse
import pickle
import time
from collections import OrderedDict, defaultdict

import opts
import models
import utils
import codecs
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='train.py')
opts.model_opts(parser)

opt = parser.parse_args()
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)
random.seed(opt.seed)
np.random.seed(opt.seed)

opts.convert_to_config(opt, config)


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


build_log()
if config.tensorboard:
    from tensorboardX import SummaryWriter
    summary_dir = config.tensorboard_log_dir + opt.log
    for file_name in os.listdir(summary_dir):
        if file_name.startswith("events.out.tfevents"):
            print(f"Event file {file_name} already exists")
            if input("Remove this file? (y/n) ") == "y":
                os.remove(os.path.join(summary_dir, file_name))
                print(f"Event file {file_name} removed")
    writer = SummaryWriter(summary_dir, comment="a")

# cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
config.use_cuda = use_cuda
if use_cuda:
    torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True


def load_data():
    print('loading data...\n')
    data = pickle.load(open(config.data + 'data.pkl', 'rb'))
    data['train']['length'] = int(data['train']['length'] * opt.scale)

    trainset = utils.BiKnowledgeDataset(
        '../data/finals/baseline/train.supporting_facts',
        infos=data['train'], char=config.char)
    validset = utils.BiKnowledgeDataset(
        '../data/finals/baseline/test.supporting_facts',
        infos=data['test'], char=config.char)

    src_vocab = data['dict']['src']
    tgt_vocab = data['dict']['tgt']
    config.src_vocab_size = src_vocab.size()
    config.tgt_vocab_size = tgt_vocab.size()

    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=0,
                                              collate_fn=utils.padding)
    if hasattr(config, 'valid_batch_size'):
        valid_batch_size = config.valid_batch_size
    else:
        valid_batch_size = config.batch_size
    validloader = torch.utils.data.DataLoader(dataset=validset,
                                              batch_size=valid_batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              collate_fn=utils.padding)

    return {'trainset': trainset, 'validset': validset,
            'trainloader': trainloader, 'validloader': validloader,
            'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab}


def build_model(checkpoints, print_log, tgt_vocab):
    for k, v in config.items():
        print_log("%s:\t%s\n" % (str(k), str(v)))

    # model
    print('building model...\n')
    model = getattr(models, config.model)(config,
                                          src_padding_idx=utils.PAD,
                                          tgt_padding_idx=utils.PAD,
                                          label_smoothing=config.label_smoothing,
                                          tgt_vocab=tgt_vocab)
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
        pre_ckpt = torch.load(opt.pretrain, map_location=lambda storage, loc: storage)['model']
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
    for k, v in config.items():
        print_log("%s:\t%s\n" % (str(k), str(v)))
    print_log("\n")
    print_log(repr(model) + "\n\n")
    print_log('total number of parameters: %d\n\n' % param_count)

    return model, optim, print_log


def train_model(model, data, optim, epoch, params):

    model.train()
    trainloader = data['trainloader']
    log_vars = defaultdict(float)

    for src, tgt, src_len, tgt_len, original_src, original_tgt, knowledge, knowledge_len in trainloader:

        model.zero_grad()

        if config.use_cuda:
            src = src.cuda()
            tgt = tgt.cuda()
            src_len = src_len.cuda()
            knowledge = knowledge.cuda()
            knowledge_len = knowledge_len.cuda()
        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        src = torch.index_select(src, dim=0, index=indices)
        tgt = torch.index_select(tgt, dim=0, index=indices)
        knowledge = torch.index_select(knowledge, dim=0, index=indices)
        knowledge_len = torch.index_select(knowledge_len, dim=0, index=indices)
        dec = tgt[:, :-1]
        targets = tgt[:, 1:]

        try:
            if config.schesamp:
                if epoch > 8:
                    e = epoch - 8
                    return_dict, outputs = model(
                        src, lengths, dec, targets, teacher_ratio=0.9**e)
                else:
                    return_dict, outputs = model(src, lengths, dec, targets)
            else:
                return_dict, outputs = model(src, lengths, dec, targets, knowledge, knowledge_len)
            pred = outputs.max(2)[1]
            targets = targets.t()
            num_correct = pred.eq(targets).masked_select(
                targets.ne(utils.PAD)).sum().item()
            num_total = targets.ne(utils.PAD).sum().item()
            if config.max_split == 0:
                return_dict['mle_loss'] = torch.sum(
                    return_dict['mle_loss']) / num_total
                if config.rl:
                    return_dict['total_loss'] = (1 - config.rl_coef) * return_dict['mle_loss'] + config.rl_coef * return_dict['rl_loss']
                else:
                    return_dict['total_loss'] = return_dict['mle_loss']
                return_dict['total_loss'].backward()
            optim.step()

            for key in return_dict:
                log_vars[key] += return_dict[key].item()
            del return_dict
            # params['report_mle_loss'] += return_dict['mle_loss'].item()
            # params['report_rl_loss'] += return_dict['rl_loss'].item()
            # params['report_total_loss'] += loss.item()
            params['report_correct'] += num_correct
            params['report_total'] += num_total

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise e

        utils.progress_bar(params['updates'], config.eval_interval)
        params['updates'] += 1

        if params['updates'] % config.report_interval == 0:
            params['log']("epoch: %3d, loss: %6.3f, time: %6.3f, updates: %8d, accuracy: %2.2f\n"
                          % (epoch, params['report_total_loss'] / config.report_interval, time.time() - params['report_time'],
                             params['updates'], params['report_correct'] * 100.0 / params['report_total']))

            if config.tensorboard:
                for key in log_vars:
                    writer.add_scalar(f"train/{key}",
                                      log_vars[key] / config.report_interval, params['updates'])
                writer.add_scalar("train" + "/lr", optim.lr, params['updates'])
                writer.add_scalar(
                    "train" + "/accuracy", params['report_correct'] / params['report_total'], params['updates'])

            log_vars = defaultdict(float)
            params['report_total_loss'], params['report_time'] = 0, time.time()
            params['report_correct'], params['report_total'] = 0, 0

        if params['updates'] % config.eval_interval == 0:
            print('evaluating after %d updates...\r' % params['updates'])
            score = eval_model(model, data, params)
            for metric in config.metrics:
                params[metric].append(score[metric])
                if score[metric] >= max(params[metric]):
                    with codecs.open(params['log_path'] + 'best_' + metric + '_prediction.txt', 'w', 'utf-8') as f:
                        f.write(codecs.open(
                            params['log_path'] + 'candidate.txt', 'r', 'utf-8').read())
                    save_model(params['log_path'] + 'best_' + metric +
                               '_checkpoint.pt', model, optim, params['updates'])
                if config.tensorboard:
                    writer.add_scalar("valid" + "/" + metric,
                                      score[metric], params['updates'])
            model.train()

        if params['updates'] % config.save_interval == 0:
            save_model(params['log_path'] + str(params['updates']) + '_checkpoint.pt',
                       model, optim, params['updates'])


def eval_model(model, data, params):

    model.eval()
    reference, candidate, source, alignments = [], [], [], []
    count, total_count = 0, len(data['validset'])
    validloader = data['validloader']
    tgt_vocab = data['tgt_vocab']

    # eval_params = defaultdict(float)
    # log_vars = defaultdict(float)
    # for src, tgt, src_len, tgt_len, original_src, original_tgt in validloader:

    #     if config.use_cuda:
    #         src = src.cuda()
    #         tgt = tgt.cuda()
    #         src_len = src_len.cuda()
    #     lengths, indices = torch.sort(src_len, dim=0, descending=True)
    #     src = torch.index_select(src, dim=0, index=indices)
    #     tgt = torch.index_select(tgt, dim=0, index=indices)
    #     dec = tgt[:, :-1]
    #     targets = tgt[:, 1:]

    #     try:
    #         with torch.no_grad():
    #             if config.schesamp:
    #                 if epoch > 8:
    #                     e = epoch - 8
    #                     return_dict, outputs = model(
    #                         src, lengths, dec, targets, teacher_ratio=0.9**e)
    #                 else:
    #                     return_dict, outputs = model(
    #                         src, lengths, dec, targets)
    #             else:
    #                 return_dict, outputs = model(src, lengths, dec, targets)
    #         pred = outputs.max(2)[1]
    #         targets = targets.t()
    #         num_correct = pred.eq(targets).masked_select(
    #             targets.ne(utils.PAD)).sum().item()
    #         num_total = targets.ne(utils.PAD).sum().item()
    #         if config.max_split == 0:
    #             return_dict['mle_loss'] = torch.sum(
    #                 return_dict['mle_loss']) / num_total
    #             if config.rl:
    #                 return_dict['total_loss'] = (1 - config.rl_coef) * return_dict['mle_loss'] + config.rl_coef * return_dict['rl_loss']
    #             else:
    #                 return_dict['total_loss'] = return_dict['mle_loss']

    #         for key in return_dict:
    #             log_vars[key] += return_dict[key].item()
    #         eval_params['report_correct'] += num_correct
    #         eval_params['report_total'] += num_total

    #     except RuntimeError as e:
    #         if 'out of memory' in str(e):
    #             print('| WARNING: ran out of memory')
    #             if hasattr(torch.cuda, 'empty_cache'):
    #                 torch.cuda.empty_cache()
    #         else:
    #             raise e

    #     eval_params['updates'] += 1

    # if config.tensorboard:
    #     for key in log_vars:
    #         writer.add_scalar(f"valid/{key}",
    #                           log_vars[key] / len(validloader), params['updates'])
    #     writer.add_scalar(
    #         "valid" + "/accuracy", eval_params['report_correct'] / eval_params['report_total'], params['updates'])

    for src, tgt, src_len, tgt_len, original_src, original_tgt, knowledge, knowledge_len in validloader:

        if config.use_cuda:
            src = src.cuda()
            src_len = src_len.cuda()
            knowledge = knowledge.cuda()
            knowledge_len = knowledge_len.cuda()

        with torch.no_grad():
            if config.beam_size > 1:
                samples, alignment, weight = model.beam_sample(
                    src, src_len, beam_size=config.beam_size, eval_=True)
            else:
                samples, alignment = model.sample(src, src_len, knowledge, knowledge_len)

        candidate += [tgt_vocab.convertToLabels(s, utils.EOS) for s in samples]
        source += original_src
        reference += original_tgt
        if alignment is not None:
            alignments += [align for align in alignment]

        count += len(original_src)
        utils.progress_bar(count, total_count)

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

    with codecs.open(params['log_path'] + 'candidate.txt', 'w+', 'utf-8') as f:
        for i in range(len(candidate)):
            f.write(" ".join(candidate[i]) + '\n')

    score = {}
    for metric in config.metrics:
        score[metric] = getattr(utils, metric)(
            reference, candidate, params['log_path'], params['log'], config)

    return score


def save_model(path, model, optim, updates):
    model_state_dict = model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'optim': optim,
        'updates': updates}
    torch.save(checkpoints, path)


def showAttention(path, s, c, attentions, index):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)
    # Set up axes
    ax.set_xticklabels([''] + s, rotation=90)
    ax.set_yticklabels([''] + c)
    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
    plt.savefig(path + str(index) + '.jpg')


def main():
    # checkpoint
    if opt.restore:
        print('loading checkpoint...\n')
        checkpoints = torch.load(opt.restore, map_location=lambda storage, loc: storage)
    else:
        checkpoints = None

    data = load_data()
    print_log, log_path = build_log()
    model, optim, print_log = build_model(checkpoints, print_log, data['tgt_vocab'])
    # scheduler
    if config.schedule:
        scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)
    params = {'updates': 0, 'report_total_loss': 0, 'report_mle_loss': 0, 'report_rl_loss': 0, 'report_reward_mean': 0, 'report_total': 0,
              'report_correct': 0, 'report_time': time.time(),
              'log': print_log, 'log_path': log_path}
    for metric in config.metrics:
        params[metric] = []
    if opt.restore:
        params['updates'] = checkpoints['updates']

    if opt.mode == "train":
        for i in range(1, config.epoch + 1):
            if config.schedule:
                scheduler.step()
                print("Decaying learning rate to %g" % scheduler.get_lr()[0])
            train_model(model, data, optim, i, params)
        for metric in config.metrics:
            print_log("Best %s score: %.2f\n" % (metric, max(params[metric])))
    else:
        score = eval_model(model, data, params)


if __name__ == '__main__':
    main()
