import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import pyrouge
import utils

# from utils.reward_provider import CTRRewardProvider


class LabelSmoothingLoss(nn.Module):

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()
        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        output = F.log_softmax(output, dim=-1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class seq2seq(nn.Module):

    def __init__(self, config, use_attention=True, encoder=None, decoder=None,
                 src_padding_idx=0, tgt_padding_idx=0, label_smoothing=0, tgt_vocab=None):
        super(seq2seq, self).__init__()

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = models.rnn_encoder(
                config, padding_idx=src_padding_idx)
        tgt_embedding = self.encoder.embedding if config.shared_vocab else None
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = models.rnn_decoder(
                config, embedding=tgt_embedding, use_attention=use_attention, padding_idx=tgt_padding_idx)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.use_cuda = config.use_cuda
        self.config = config
        self.label_smoothing = label_smoothing
        if self.label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                label_smoothing, config.tgt_vocab_size,
                ignore_index=tgt_padding_idx)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=utils.PAD, reduction='none')
        if config.use_cuda:
            self.criterion.cuda()
        if config.rl:
            # self.reward_provider = CTRRewardProvider(
            #     config.ctr_rewared_provider_path)
            self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_padding_idx

    def compute_loss(self, scores, targets):
        scores = scores.view(-1, scores.size(2))
        loss = self.criterion(scores, targets.contiguous().view(-1))
        return loss

    def compute_reward(self, tgt, state):
        sample_ids, probs, entropy = self.rl_sample(state)
        sample_ids = sample_ids.t().tolist()
        probs = probs.t()
        for i, sample_id in enumerate(sample_ids):
            x = sample_id.index(3) + 1 if 3 in sample_id else len(sample_id)
            probs[i][x:] = 0
            # probs[i] /= x  # length norm
        tgt = tgt.tolist()
        batch_size = probs.size(0)
        # rewards = self.reward_func_batch(sample_ids, tgt)
        rewards = []
        for y, y_hat in zip(sample_ids, tgt):
            rewards.append(self.reward_func(y, y_hat))
        rewards = torch.tensor(rewards).unsqueeze(1).expand_as(probs).cuda()

        if self.config.baseline == 'self_critic':
            # for baseline
            with torch.no_grad():
                greedy_pred = self.greedy_sample(state)
            # baselines = self.reward_func_batch(greedy_pre, tgt)
            baselines = []
            for y, y_hat in zip(greedy_pred, tgt):
                baselines.append(self.reward_func(y, y_hat))
            baselines = torch.tensor(baselines).unsqueeze(
                1).expand_as(probs).cuda()
            rewards = rewards - baselines
        loss = -probs * rewards
        return loss, rewards, baselines, entropy

    def reward_func(self, y, y_hat, func='mlc'):
        """Define your own reward function. Predefined functions are mlc, bleu, rouge"""
        # hypo = self.tgt_vocab.convertToLabels(y, utils.EOS)
        # target = self.tgt_vocab.convertToLabels(y_hat, utils.EOS)
        # reward = self.reward_provider.reward_fn(hypo, target)
        if func == 'mlc':
            y_true = np.zeros(self.config.tgt_vocab_size - 4)
            y_pre = np.zeros(self.config.tgt_vocab_size - 4)
            for i in y:
                if i == 3:
                    break
                else:
                    if i > 3:
                        y_true[i - 4] = 1
            for i in y_hat:
                if i == 3:
                    break
                else:
                    if i > 3:
                        y_pre[i - 4] = 1
            if self.config.reward == 'f1':
                reward = utils.metrics.f1_score(
                    np.array([y_true]), np.array([y_pre]), average='micro')
            elif self.config.reward == 'hamming_loss':
                reward = metrics.hamming_loss(
                    np.array([y_true]), np.array([y_pre]))
        return reward

    # def reward_func_batch(self, ys, y_hats):
    #     hypos = [self.tgt_vocab.convertToLabels(y, utils.EOS) for y in ys]
    #     targets = [self.tgt_vocab.convertToLabels(
    #         y_hat, utils.EOS) for y_hat in y_hats]
    #     reward = self.reward_provider.reward_fn_batched(hypos, targets)
    #     return reward

    def greedy_sample(self, state):
        bos = torch.ones(state[0].size(1)).long().fill_(utils.BOS).cuda()
        inputs, outputs, attn_matrix = [bos], [], []

        for i in range(self.config.max_time_step):
            output, state, attn_weights = self.decoder(inputs[i], state)
            predicted = output.max(1)[1]
            inputs += [predicted]
            outputs += [predicted]
            attn_matrix += [attn_weights]
        sample_ids = torch.stack(outputs).t().tolist()

        return sample_ids

    def rl_sample(self, state):
        bos = torch.ones(state[0].size(1)).long().fill_(utils.BOS).cuda()
        inputs = [bos]
        sample_ids, probs = [], []

        entropy = 0
        for i in range(self.config.max_time_step):
            output, state, attn_weights = self.decoder(inputs[i], state)
            entropy += (-F.softmax(output, dim=-1) * F.log_softmax(output, dim=-1)).sum(dim=1)
            predicted = F.softmax(output, dim=-1).multinomial(1).squeeze()  # [batch]
            prob = F.log_softmax(output, dim=-1)[range(len(predicted)), predicted]
            inputs += [predicted]
            sample_ids += [predicted]
            probs += [prob]

        entropy /= self.config.max_time_step
        sample_ids = torch.stack(sample_ids).squeeze()  # [max_tgt_len, batch]
        probs = torch.stack(probs).squeeze()  # [max_tgt_len, batch]

        return sample_ids, probs, entropy

    def forward(self, src, src_len, dec, targets, teacher_ratio=1.0):
        return_dict = {}
        src = src.t()
        dec = dec.t()
        tgt = targets
        targets = targets.t()
        teacher = random.random() < teacher_ratio

        contexts, state = self.encoder(src, src_len.tolist())

        if self.decoder.attention is not None:
            self.decoder.attention.init_context(context=contexts)

        if self.config.rl:
            rl_loss, rewards, baselines, entropy = self.compute_reward(
                tgt, state)
            # TODO entropy
            rl_loss = rl_loss
            return_dict['rl_loss'] = rl_loss.sum(dim=1).mean()
            return_dict['reward_mean'] = rewards[:, 0].mean()
            return_dict['greedy_mean'] = baselines[:, 0].mean()
            return_dict['sample_mean'] = return_dict['reward_mean'] + \
                return_dict['greedy_mean']
            return_dict['entropy'] = entropy.mean()

        outputs = []
        if teacher:
            for input in dec.split(1):
                output, state, attn_weights = self.decoder(
                    input.squeeze(0), state)
                outputs.append(output)
            outputs = torch.stack(outputs)
        else:
            inputs = [dec.split(1)[0].squeeze(0)]
            for i, _ in enumerate(dec.split(1)):
                output, state, attn_weights = self.decoder(inputs[i], state)
                predicted = output.max(1)[1]
                inputs += [predicted]
                outputs.append(output)
            outputs = torch.stack(outputs)

        loss = self.compute_loss(outputs, targets)

        return_dict['mle_loss'] = loss

        return return_dict, outputs

    def sample(self, src, src_len):

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, reverse_indices = torch.sort(indices)
        src = torch.index_select(src, dim=0, index=indices)
        bos = torch.ones(src.size(0)).long().fill_(utils.BOS).cuda()
        src = src.t()

        contexts, state = self.encoder(src, lengths.tolist())

        if self.decoder.attention is not None:
            self.decoder.attention.init_context(context=contexts)
        inputs, outputs, attn_matrix = [bos], [], []
        for i in range(self.config.max_time_step):
            output, state, attn_weights = self.decoder(inputs[i], state)
            predicted = output.max(1)[1]
            inputs += [predicted]
            outputs += [predicted]
            attn_matrix += [attn_weights]

        outputs = torch.stack(outputs)
        sample_ids = torch.index_select(
            outputs, dim=1, index=reverse_indices).t().tolist()

        if self.decoder.attention is not None:
            attn_matrix = torch.stack(attn_matrix)
            alignments = attn_matrix.max(2)[1]
            alignments = torch.index_select(
                alignments, dim=1, index=reverse_indices).t().tolist()
        else:
            alignments = None

        return sample_ids, alignments, attn_matrix

    def beam_sample(self, src, src_len, beam_size=1, eval_=False):

        # (1) Run the encoder on the src.

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, ind = torch.sort(indices)
        src = torch.index_select(src, dim=0, index=indices)
        src = src.t()
        batch_size = src.size(1)
        contexts, encState = self.encoder(src, lengths.tolist())

        #  (1b) Initialize for the decoder.

        def rvar(a):
            return a.repeat(1, beam_size, 1)

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # Repeat everything beam_size times.
        # contexts = rvar(contexts.data)
        contexts = rvar(contexts)

        if self.config.cell == 'lstm':
            decState = (rvar(encState[0]), rvar(encState[1]))
        else:
            decState = rvar(encState)

        beam = [models.Beam(beam_size, n_best=1,
                            cuda=self.use_cuda, length_norm=self.config.length_norm)
                for __ in range(batch_size)]
        if self.decoder.attention is not None:
            self.decoder.attention.init_context(contexts)

        # (2) run the decoder to generate sentences, using beam search.

        for i in range(self.config.max_time_step):

            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = torch.stack([b.getCurrentState()
                               for b in beam]).t().contiguous().view(-1)

            # Run one step.
            output, decState, attn = self.decoder(inp, decState)
            # decOut: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            output = unbottle(self.log_softmax(output))
            attn = unbottle(attn)
            # beam x tgt_vocab

            # (c) Advance each beam.
            # update state
            for j, b in enumerate(beam):
                b.advance(output[:, j], attn[:, j])
                if self.config.cell == 'lstm':
                    b.beam_update(decState, j)
                else:
                    b.beam_update_gru(decState, j)

        # (3) Package everything up.
        allHyps, allScores, allAttn = [], [], []
        if eval_:
            allWeight = []

        for j in ind:
            b = beam[j]
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            if eval_:
                weight = []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att.max(1)[1])
                if eval_:
                    weight.append(att)
            allHyps.append(hyps[0])
            allScores.append(scores[0])
            allAttn.append(attn[0])
            if eval_:
                allWeight.append(weight[0])

        if eval_:
            return allHyps, allAttn, allWeight

        return allHyps, allAttn
