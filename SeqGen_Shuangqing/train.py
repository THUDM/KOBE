from matplotlib import pyplot as plt
from matplotlib import ticker
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
# import sys
import os
import json
import lr_scheduler as L
from torch.nn.init import xavier_uniform_
import torch.utils.data
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')

# parser for the introduction of hyperparameters
parser = argparse.ArgumentParser(description='train.py')
opts.model_opts(parser)
opt = parser.parse_args()
config = utils.read_config(opt.config)
opts.convert_to_config(opt, config)
if opt.label_dict_file:
    with open(opt.label_dict_file, 'r') as f:
        label_dict = json.load(f)

# seed
torch.manual_seed(opt.seed)
random.seed(opt.seed)
np.random.seed(opt.seed)


# build log, including tensorboard
def build_log():
    """
    build log.
    :return: print_log function and log path
    """
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
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
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
    # torch.cuda.set_device(opt.gpus[0])
    if len(opt.gpus) > 1:
        torch.cuda.manual_seed_all(opt.seed)
    else:
        torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
device = torch.device('cuda:{}'.format(opt.gpus[0])) if use_cuda else torch.device('cpu')
devices_ids = opt.gpus


# load data
def load_data():
    """
    load data.
    update "data" due to the saved path in the pickle file
    :return: a dict with data and vocabulary
    """
    print('loading data...\n')
    data = pickle.load(open(config.data + 'data.pkl', 'rb'))
    # retrieve data, due to the problem of path.
    data['train']['length'] = int(data['train']['length'] * opt.scale)
    data['train']['srcF'] = os.path.join(config.data, 'train.src.id')
    data['train']['original_srcF'] = os.path.join(config.data, 'train.src.str')
    data['train']['tgtF'] = os.path.join(config.data, 'train.tgt.id')
    data['train']['original_tgtF'] = os.path.join(config.data, 'train.tgt.str')
    data['test']['srcF'] = os.path.join(config.data, 'test.src.id')
    data['test']['original_srcF'] = os.path.join(config.data, 'test.src.str')
    data['test']['tgtF'] = os.path.join(config.data, 'test.tgt.id')
    data['test']['original_tgtF'] = os.path.join(config.data, 'test.tgt.str')

    train_set = utils.BiDataset(data['train'], char=config.char)
    valid_set = utils.BiDataset(data['test'], char=config.char)

    src_vocab = data['dict']['src']
    tgt_vocab = data['dict']['tgt']
    config.src_vocab_size = src_vocab.size()
    config.tgt_vocab_size = tgt_vocab.size()

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               collate_fn=utils.padding)
    if hasattr(config, 'valid_batch_size'):
        valid_batch_size = config.valid_batch_size
    else:
        valid_batch_size = config.batch_size
    valid_loader = torch.utils.data.DataLoader(dataset=valid_set,
                                               batch_size=valid_batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               collate_fn=utils.padding)
    return {'train_set': train_set, 'valid_set': valid_set,
            'train_loader': train_loader, 'valid_loader': valid_loader,
            'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab}


# build model
def build_model(checkpoints, print_log, tgt_vocab):
    """
    build model, either Seq2Seq or Tensor2Tensor
    :param checkpoints: load checkpoint if there is pretrained model
    :param print_log: function to print log
    :param tgt_vocab: target-side vocabulary
    :return: model, optimizer and the print_log function
    """
    for k, v in config.items():
        print_log("%s:\t%s\n" % (str(k), str(v)))

    # model
    print('building model...\n')
    model = getattr(models, config.model)(config,
                                          src_padding_idx=utils.PAD,
                                          tgt_padding_idx=utils.PAD,
                                          label_smoothing=config.label_smoothing,
                                          tgt_vocab=None)
    model.to(device)
    if len(opt.gpus) > 1:
        # TODO: distributed training, now only data parallel
        # torch.distributed.init_process_group(backend="nccl")
        # model.to(device)
        model = nn.DataParallel(model, device_ids=devices_ids, output_device=device)
        if config.param_init != 0.0:
            for p in model.module.parameters():
                p.data.uniform_(-config.param_init, config.param_init)
        if config.param_init_glorot:
            for p in model.module.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
    else:
        if config.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-config.param_init, config.param_init)
        if config.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
    if checkpoints is not None:
        if len(opt.gpus) > 1:
            model.module.load_state_dict(checkpoints['model'])
        else:
            model.load_state_dict(checkpoints['model'])
    if opt.pretrain:
        print('loading checkpoint from %s' % opt.pretrain)
        pre_ckpt = torch.load(
            opt.pretrain, map_location=lambda storage, loc: storage)['model']
        if len(opt.gpus) > 1:
            model.module.load_state_dict(pre_ckpt)
        else:
            model.load_state_dict(pre_ckpt)
        # pre_ckpt = OrderedDict({key[8:]: pre_ckpt[key]
        #                         for key in pre_ckpt if key.startswith('encoder')})
        # print(model.encoder.state_dict().keys())
        # print(pre_ckpt.keys())
        # model.encoder.load_state_dict(pre_ckpt)

    # optimizer
    optim = models.Optim(config.optim,
                         config.learning_rate, config.max_grad_norm,
                         lr_decay=config.learning_rate_decay,
                         start_decay_steps=config.start_decay_steps,
                         beta1=config.beta1, beta2=config.beta2,
                         decay_method=config.decay_method,
                         warmup_steps=config.warmup_steps,
                         model_size=config.hidden_size)
    if checkpoints is not None:
        optim = optim.optimizer.load_state_dict(checkpoints['optim'])
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


# train model
def train_model(model, data, optim, epoch, params):
    """
    model training
    :param model: model
    :param data: data
    :param optim: optimizer
    :param epoch: total epochs
    :param params: parameters
    :return: none
    """
    model.train()
    train_loader = data['train_loader']
    log_vars = defaultdict(float)

    for src, tgt, src_len, tgt_len, original_src, original_tgt in train_loader:
        # put the tensors on cuda devices
        # device = torch.device(":0")
        src, tgt = src.to(device), tgt.to(device)
        src_len, tgt_len = src_len.to(device), tgt_len.to(device)
        # original_src, original_tgt = original_src.to(device), original_tgt.to(device)

        # remove the gradients
        model.zero_grad()

        # reverse sort the lengths for rnn
        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        # select by the indices
        src = torch.index_select(src, dim=0, index=indices)  # [batch, len]
        tgt = torch.index_select(tgt, dim=0, index=indices)  # [batch, len]
        dec = tgt[:, :-1]   # [batch, len]
        targets = tgt[:, 1:]    # [batch, len]

        try:
            if config.schesamp:
                if epoch > 8:
                    e = epoch - 8
                    return_dict, outputs = model(
                        src, lengths, dec, targets, teacher_ratio=0.9**e)
                else:
                    return_dict, outputs = model(src, lengths, dec, targets)
            else:
                return_dict, outputs = model(src, lengths, dec, targets)    # outputs: [batch, len, size]
            pred = outputs.transpose(0, 1).max(2)[1]
            targets = targets.t()
            num_correct = pred.eq(targets).masked_select(
                targets.ne(utils.PAD)).sum().item()
            num_total = targets.ne(utils.PAD).sum().item()

            return_dict['mle_loss'] = torch.sum(
                return_dict['mle_loss']) / num_total
            if config.rl:
                return_dict['total_loss'] = return_dict['mle_loss'] + \
                    config.rl_coef * return_dict['rl_loss']
            else:
                return_dict['total_loss'] = return_dict['mle_loss']
            return_dict['total_loss'].backward()
            optim.step()

            for key in return_dict:
                log_vars[key] += return_dict[key].item()
            params['report_total_loss'] += return_dict['total_loss'].item()
            params['report_correct'] += num_correct
            params['report_total'] += num_total

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise e

        # utils.progress_bar(params['updates'], config.eval_interval)
        params['updates'] += 1

        if params['updates'] % config.report_interval == 0:
            params['log']("epoch: %3d, loss: %6.3f, time: %6.3f, updates: %8d, accuracy: %2.2f\n"
                          % (epoch,
                             params['report_total_loss'] /
                             config.report_interval,
                             time.time() -
                             params['report_time'], params['updates'],
                             params['report_correct'] * 100.0 / params['report_total']))

            if config.tensorboard:
                for key in return_dict:
                    writer.add_scalar(f"train/{key}",
                                      log_vars[key] / config.report_interval,
                                      params['updates'])
                # writer.add_scalar("train" + "/lr", optim.lr, params['updates'])
                writer.add_scalar(
                    "train" +
                    "/accuracy", params['report_correct'] /
                    params['report_total'],
                    params['updates'])

            log_vars = defaultdict(float)
            params['report_total_loss'], params['report_time'] = 0, time.time()
            params['report_correct'], params['report_total'] = 0, 0

        if params['updates'] % config.eval_interval == 0:
            print('evaluating after %d updates...\r' % params['updates'])
            score = eval_model(model, data, params)
            for metric in config.metrics:
                params[metric].append(score[metric])
                if score[metric] >= max(params[metric]):
                    with codecs.open(params['log_path'] + 'best_' + metric + '_prediction.txt',
                                     'w', 'utf-8') as f:
                        f.write(codecs.open(
                            params['log_path'] + 'candidate.txt', 'r', 'utf-8').read())
                    save_model(params['log_path'] + 'best_' + metric +
                               '_checkpoint.pt', model, optim, params['updates'])
                if config.tensorboard:
                    writer.add_scalar("valid" + "/" + metric,
                                      score[metric], params['updates'])
            model.train()

        if params['updates'] % config.save_interval == 0:
            if config.save_individual:
                save_model(params['log_path'] + str(params['updates']) + 'checkpoint.pt',
                           model, optim, params['updates'])
            save_model(params['log_path'] + 'checkpoint.pt',
                       model, optim, params['updates'])

    if config.epoch_decay:
        if len(opt.gpus) > 1:
            optim.module.updateLearningRate(epoch)
        else:
            optim.updateLearningRate(epoch)


# evaluate model
def eval_model(model, data, params):
    """
    model evaluation
    :param model: model
    :param data: data
    :param params: parameters
    :return: evaluation scores
    """
    model.eval()
    reference, candidate, source, alignments = [], [], [], []
    # count, total_count = 0, len(data['valid_set'])
    valid_loader = data['valid_loader']
    tgt_vocab = data['tgt_vocab']

    # eval_params = defaultdict(float)
    # log_vars = defaultdict(float)
    # for src, tgt, src_len, tgt_len, original_src, original_tgt in valid_loader:

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
    #                 return_dict['total_loss'] = return_dict['mle_loss'] + \
    #                     return_dict['rl_loss'] * config.rl_coef
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
    #     for key in return_dict:
    #         writer.add_scalar(f"valid/{key}",
    #                           log_vars[key] / len(valid_loader), params['updates'])
    #     writer.add_scalar(
    #         "valid" + "/accuracy",
    #         eval_params['report_correct'] / eval_params['report_total'],
    #         params['updates'])

    for src, tgt, src_len, tgt_len, original_src, original_tgt in valid_loader:
        src = src.to(device)
        src_len = src_len.to(device)

        with torch.no_grad():
            if config.beam_size > 1:
                samples, alignment, weight = model.beam_sample(
                    src, src_len, beam_size=config.beam_size, eval_=True)
            else:
                samples, alignment = model.sample(src, src_len)

        candidate += [tgt_vocab.convertToLabels(s, utils.EOS) for s in samples]
        source += original_src
        reference += original_tgt
        if alignment is not None:
            alignments += [align for align in alignment]

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


# save model
def save_model(path, model, optim, updates):
    if len(opt.gpus) > 1:
        model_state_dict = model.module.state_dict()
        optim_state_dict = optim.optimizer.state_dict()
    else:
        model_state_dict = model.state_dict()
        optim_state_dict = optim.optimizer.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'updates': updates,
        'optim': optim_state_dict}
    torch.save(checkpoints, path)


# attention visualization
def showAttention(path, s, c, attentions, index):
    """
    attention visualization
    :param path: saved path
    :param s: source
    :param c: candidate
    :param attentions: attention scores
    :param index: instance index
    :return: none
    """
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


# main function
def main():
    """
    main function for execution
    :return: none
    """
    # checkpoint
    if opt.restore:
        print('loading checkpoint...\n')
        checkpoints = torch.load(
            opt.restore, map_location=lambda storage, loc: storage)
    else:
        checkpoints = None
    # load data
    data = load_data()
    print_log, log_path = build_log()
    model, optim, print_log = build_model(
        checkpoints, print_log, data['tgt_vocab'])
    # scheduler
    if config.schedule:
        scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)
    # parameters
    params = {'updates': 0, 'report_total_loss': 0, 'report_total': 0,
              'report_correct': 0, 'report_time': time.time(),
              'log': print_log, 'log_path': log_path}
    for metric in config.metrics:
        params[metric] = []
    if opt.restore:
        params['updates'] = checkpoints['updates']
    # train or evaluate
    if opt.mode == "train":
        for i in range(1, config.epoch + 1):
            if config.schedule:
                scheduler.step()
                print("Decaying learning rate to %g" % scheduler.get_lr()[0])
            train_model(model, data, optim, i, params)
        for metric in config.metrics:
            print_log("Best %s score: %.3f\n" % (metric, max(params[metric])))
    else:
        score = eval_model(model, data, params)


if __name__ == '__main__':
    main()
