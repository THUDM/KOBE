import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import models
import random
from fairseq import bleu
from utils.reward_provider import CTRRewardProvider


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


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


class tensor2tensor(nn.Module):

    def __init__(self, config, use_attention=True,
                 encoder=None, decoder=None,
                 src_padding_idx=0, tgt_padding_idx=0,
                 label_smoothing=0, tgt_vocab=None):
        super(tensor2tensor, self).__init__()

        self.config = config

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = models.TransformerEncoder(
                config, padding_idx=src_padding_idx)
        tgt_embedding = self.encoder.embedding if config.shared_vocab else None
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = models.TransformerDecoder(
                config, tgt_embedding=tgt_embedding, padding_idx=tgt_padding_idx)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.use_cuda = config.use_cuda
        self.config = config
        self.label_smoothing = label_smoothing
        if self.label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                label_smoothing, config.tgt_vocab_size,
                ignore_index=tgt_padding_idx)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=utils.PAD)
        if config.use_cuda:
            self.criterion.cuda()
        self.compute_score = nn.Linear(config.hidden_size, config.tgt_vocab_size)
        if config.rl:
            self.bleu_scorer = bleu.Scorer(pad=0, eos=3, unk=1)
            self.reward_provider = CTRRewardProvider(
                config.ctr_rewared_provider_path)
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_padding_idx
        self.transform_embedding = nn.Linear(config.emb_size, config.emb_size // 2)
        self.fact_embedding = nn.Embedding(config.tgt_vocab_size, config.emb_size // 2,
                                           padding_idx=tgt_padding_idx)

    def compute_loss(self, scores, targets):
        scores = scores.contiguous().view(-1, scores.size(2))
        loss = self.criterion(scores, targets.contiguous().view(-1))
        return loss

    def compute_reward(self, src, contexts, tgt):
        batch_size = src.size(0)
        tgt = tgt.t().tolist()
        # src: max_len X batch
        # tgt: batch X max_len
        # sample_ids: batch X max_len
        sample_ids, probs, entropy = self.rl_sample(src, contexts)
        sample_ids = sample_ids.tolist()
        sample_lens = []
        for i, sample_id in enumerate(sample_ids):
            x = sample_id.index(3) + 1 if 3 in sample_id else len(sample_id)
            probs[i][x:] = 0
            sample_lens.append(x)
            # probs[i] /= x  # length norm
        rewards = self.reward_func_batch(sample_ids, tgt, sample_lens)
        # rewards = torch.Tensor(rewards).cuda()

        # rewards = []
        # for y, y_hat in zip(sample_ids, tgt):
        #     rewards.append(self.reward_func(y, y_hat))
        # rewards = torch.tensor(rewards).unsqueeze(1).expand_as(probs).cuda()

        if self.config.baseline == 'self_critic':
            # for baseline
            with torch.no_grad():
                greedy_pre = self.greedy_sample(src, contexts)
            baselines = self.reward_func_batch(greedy_pre, tgt, sample_lens)
            # baselines = torch.Tensor(baselines).cuda()
            # baselines = []
            # for y, y_hat in zip(greedy_pre, tgt):
            #     baselines.append(self.reward_func(y, y_hat))
            # baselines = torch.tensor(baselines).unsqueeze(
            # 1).expand_as(probs).cuda()
            rewards = rewards - baselines
        else:
            baselines = torch.zeros(batch_size, 1).cuda()

        loss = -probs * rewards
        # rewards /= sample_lens
        # baselines /= sample_lens
        # elif self.config.reward == 'hamming_loss':
        # loss = (probs * rewards)
        # Length normalization
        sample_lens = torch.Tensor(sample_lens).cuda().unsqueeze(1)
        return loss, rewards / sample_lens * self.config.max_time_step,\
            baselines / sample_lens * self.config.max_time_step, entropy

    def reward_func_batch(self, ys, y_hats, sample_lens):
        EOS_TOKEN = 3
        if self.config.rl_reward_type == 'bleu':
            batch_rewards = []
            for hypo, target, sample_len in zip(ys, y_hats, sample_lens):
                if EOS_TOKEN in hypo:
                    hypo = hypo[:hypo.index(EOS_TOKEN) + 1]
                if EOS_TOKEN in target:
                    target = target[:target.index(EOS_TOKEN) + 1]
                if self.config.rl_stepwise:
                    reward = []
                    for step in range(sample_len):
                        self.bleu_scorer.reset()
                        self.bleu_scorer.add(torch.IntTensor(hypo[:step + 1]),
                                            torch.IntTensor(target))
                        reward.append(self.bleu_scorer.score())
                    batch_rewards.append(
                        reward + [0] * (self.config.max_time_step - sample_len))
                else:
                    self.bleu_scorer.reset()
                    self.bleu_scorer.add(torch.IntTensor(
                        hypo), torch.IntTensor(target))
                    batch_rewards.append(self.bleu_scorer.score())
            if self.config.rl_stepwise:
                batch_rewards = torch.Tensor(batch_rewards).cuda()
            else:
                batch_rewards = torch.tensor(batch_rewards).unsqueeze(
                    1).expand(-1, self.config.max_time_step).cuda()
        elif self.config.rl_reward_type == 'ctr':
            assert not self.config.rl_stepwise
            hypos = [self.tgt_vocab.convertToLabels(y, utils.EOS) for y in ys]
            targets = [self.tgt_vocab.convertToLabels(y_hat, utils.EOS) for y_hat in y_hats]
            batch_rewards = self.reward_provider.reward_fn_batched(hypos, targets)
            batch_rewards = torch.tensor(batch_rewards).unsqueeze(
                1).expand(-1, self.config.max_time_step).cuda()
        return batch_rewards

    def forward(self, src, src_len, dec, targets, knowledge, knowledge_len, teacher_ratio=1.0):
        return_dict = {}
        src = src.t()
        dec = dec.t()
        knowledge = knowledge.t()
        targets = targets.t()

        # MLE Loss
        outputs = []
        if self.config.positional:
            # knowledge_embed = self.transform_embedding(self.decoder.embedding(knowledge))
            knowledge_embed = self.fact_embedding(knowledge)
            # knowledge_embed = self.fact_embedding(knowledge)
            contexts = self.encoder(src, knowledge_embed, src_len.tolist())
            # mask = (knowledge != 0).float()
            # knowledge_embed = self.knowledge_embedding(knowledge)
            # contexts = self.encoder.condition_context_attn(contexts.transpose(0, 1), knowledge_embed, mask).transpose(0, 1)
            self.decoder.init_state(src, contexts)
            outputs, _ = self.decoder(dec, contexts)
        else:
            contexts, state = self.encoder(src, src_len.tolist())
            self.decoder.init_state(src, contexts)
            outputs, _, state = self.decoder(dec, contexts, state)
        scores = self.compute_score(outputs.transpose(0, 1)).transpose(0, 1)
        loss = self.compute_loss(scores, targets)
        return_dict['mle_loss'] = loss

        # Policy Gradient
        if self.config.rl:
            rl_loss, rewards, baselines, entropy = self.compute_reward(
                src, contexts, targets)
            return_dict['rl_loss'] = rl_loss.sum(dim=1).mean()
            return_dict['reward_mean'] = rewards.mean()
            return_dict['greedy_mean'] = baselines.mean()
            return_dict['sample_mean'] = return_dict['reward_mean'] + \
                return_dict['greedy_mean']
            return_dict['entropy'] = entropy.mean()

        return return_dict, scores

    def sample(self, src, src_len, knowledge, knowledge_len):
        # lengths, indices = torch.sort(src_len, dim=0, descending=True)
        # _, reverse_indices = torch.sort(indices)
        # src = torch.index_select(src, dim=0, index=indices)
        bos = torch.ones(src.size(0)).long().fill_(utils.BOS)
        if self.use_cuda:
            bos = bos.cuda()
        src = src.t()
        knowledge = knowledge.t()

        if self.config.positional:
            # knowledge_embed = self.transform_embedding(self.decoder.embedding(knowledge))
            knowledge_embed = self.fact_embedding(knowledge)
            contexts = self.encoder(src, knowledge_embed, src_len.tolist())
            # contexts = self.encoder(src, knowledge, src_len.tolist())
            # mask = (knowledge != 0).float()
            # knowledge_embed = self.knowledge_embedding(knowledge)
            # contexts = self.encoder.condition_context_attn(contexts.transpose(0, 1), knowledge_embed, mask).transpose(0, 1)
        else:
            contexts, state = self.encoder(src, src_len.tolist())

        self.decoder.init_state(src, contexts)
        inputs, outputs, attn_matrix = [bos], [], []
        for i in range(self.config.max_time_step):
            if self.config.positional:
                output, attn_weights = self.decoder(
                    inputs[i].unsqueeze(0), contexts, step=i)
            else:
                output, attn_weights, state = self.decoder(
                    inputs[i].unsqueeze(0), contexts, state, step=i)
            output = self.compute_score(output.transpose(0, 1)).squeeze(1)
            predicted = output.max(1)[1]
            inputs.append(predicted)
            outputs.append(predicted)
            attn_matrix.append(attn_weights.squeeze(0))
        outputs = torch.stack(outputs)
        sample_ids = outputs.t()
        # sample_ids = torch.index_select(
        # outputs, dim=1, index=reverse_indices).t()

        attn_matrix = torch.stack(attn_matrix)
        alignments = attn_matrix.max(2)[1]
        alignments = alignments.t()
        # alignments = torch.index_select(
        #     alignments, dim=1, index=reverse_indices).t()

        return sample_ids.tolist(), alignments.tolist()

    def rl_sample(self, src, contexts):
        # src: max_len X batch
        bos = torch.ones(src.size(1)).long().fill_(utils.BOS).cuda()
        self.decoder.init_state(src, contexts)
        inputs, outputs = [bos], []
        probs = []
        entropy = 0
        for i in range(self.config.max_time_step):
            assert self.config.positional
            output, _ = self.decoder(inputs[i].unsqueeze(0), contexts, step=i)
            output = self.compute_score(output.transpose(0, 1)).squeeze(1)
            entropy += (-F.softmax(output) *
                        F.log_softmax(output)).sum(dim=1).data
            predicted = F.softmax(output).multinomial(1).squeeze()
            prob = F.log_softmax(output)[range(len(predicted)), predicted]
            probs += [prob]
            inputs.append(predicted)
            outputs.append(predicted)
        # [batch, max_len]
        sample_ids = torch.stack(outputs).t()
        probs = torch.stack(probs).squeeze().t()
        entropy /= self.config.max_time_step

        return sample_ids, probs, entropy

    def greedy_sample(self, src, contexts):
        bos = torch.ones(src.size(1)).long().fill_(utils.BOS).cuda()
        self.decoder.init_state(src, contexts)
        inputs, outputs = [bos], []
        for i in range(self.config.max_time_step):
            assert self.config.positional
            output, _ = self.decoder(inputs[i].unsqueeze(0), contexts, step=i)
            output = self.compute_score(output.transpose(0, 1)).squeeze(1)
            predicted = output.max(1)[1]
            inputs.append(predicted)
            outputs.append(predicted)
        sample_ids = torch.stack(outputs).t().tolist()

        return sample_ids

    def beam_sample(self, src, src_len, beam_size=1, eval_=False):

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, ind = torch.sort(indices)
        src = torch.index_select(src, dim=0, index=indices)
        src = src.t()
        batch_size = src.size(1)

        if self.config.positional:
            contexts = self.encoder(src, lengths.tolist())
        else:
            contexts, state = self.encoder(src, lengths.tolist())

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(batch_size, beam_size, -1)

        beam = [models.Beam(beam_size, n_best=1,
                            cuda=self.use_cuda, length_norm=self.config.length_norm)
                for __ in range(batch_size)]

        contexts = tile(contexts, beam_size, 1)
        src = tile(src, beam_size, 1)

        if not self.config.positional:
            h = tile(state[0], beam_size, 0)
            c = tile(state[1], beam_size, 0)
            state = (h, c)

        self.decoder.init_state(src, contexts)

        for i in range(self.config.max_time_step):

            if all((b.done() for b in beam)):
                break

            inp = torch.stack([b.getCurrentState() for b in beam])
            inp = inp.view(1, -1)

            if self.config.positional:
                output, attn = self.decoder(inp, contexts, step=i)
                state = None
            else:
                output, attn, state = self.decoder(
                    inp, contexts, state, step=i)
            output = self.compute_score(output.transpose(0, 1)).squeeze(1)

            output = unbottle(self.log_softmax(output))
            attn = unbottle(attn.squeeze(0))

            select_indices_array = []
            for j, b in enumerate(beam):
                b.advance(output[j, :], attn[j, :])
                select_indices_array.append(
                    b.getCurrentOrigin() + j * beam_size)
            select_indices = torch.cat(select_indices_array)
            self.decoder.map_state(
                lambda state, dim: state.index_select(dim, select_indices))
            if state is not None:
                state = (state[0].index_select(0, select_indices),
                         state[1].index_select(0, select_indices))

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
