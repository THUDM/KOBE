import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils

# from fairseq import bleu
# from utils.reward_provider import CTRRewardProvider


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
    """ Label smoothing loss """
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
    """ transformer model """
    def __init__(self, config, use_attention=True,
                 encoder=None, decoder=None,
                 src_padding_idx=0, tgt_padding_idx=0,
                 label_smoothing=0, tgt_vocab=None):
        """
        Initialization of variables and functions
        :param config: configuration
        :param use_attention: use attention or not, consistent with seq2seq
        :param encoder: encoder
        :param decoder: decoder
        :param src_padding_idx: source padding index
        :param tgt_padding_idx: target padding index
        :param label_smoothing: ratio for label smoothing
        :param tgt_vocab: target vocabulary
        """
        super(tensor2tensor, self).__init__()

        self.config = config

        # pretrained encoder or not
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = models.TransformerEncoder(
                config, padding_idx=src_padding_idx)
        tgt_embedding = self.encoder.embedding if config.shared_vocab else None
        # pretrained decoder or not
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = models.TransformerDecoder(
                config, tgt_embedding=tgt_embedding, padding_idx=tgt_padding_idx)
        # log softmax should specify dimension explicitly
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
        self.compute_score = nn.Linear(
            config.hidden_size, config.tgt_vocab_size)

        # Use rl or not. Should specify a reward provider. Not available yet in this framework.
        # if config.rl:
            # self.bleu_scorer = bleu.Scorer(pad=0, eos=3, unk=1)
            # self.reward_provider = CTRRewardProvider(config.ctr_reward_provider_path)
            # self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_padding_idx

    def compute_loss(self, scores, targets):
        """
        loss computation
        :param scores: predicted scores
        :param targets: targets
        :return: loss
        """
        scores = scores.contiguous().view(-1, scores.size(2))   #[len*batch, vocab]
        loss = self.criterion(scores, targets.contiguous().view(-1))
        return loss

    def compute_reward(self, contexts, tgt):
        """
        Reward computation
        :param contexts: source contexts
        :param tgt: target
        :return: loss, rewards, baselines and entropy
        """
        sample_ids, probs, entropy = self.rl_sample(contexts)
        sample_ids = sample_ids
        probs = probs.t()
        for i, sample_id in enumerate(sample_ids):
            x = sample_id.index(3) + 1 if 3 in sample_id else len(sample_id)
            probs[i][x:] = 0
            probs[i] /= x  # length norm
        tgt = tgt.tolist()
        # rewards = self.reward_func_batch(sample_ids, tgt)
        rewards = []
        for y, y_hat in zip(sample_ids, tgt):
            rewards.append(self.reward_func(y, y_hat))
        rewards = torch.tensor(rewards).unsqueeze(1).expand_as(probs).cuda()

        if self.config.baseline == 'self_critic':
            # for baseline
            greedy_pred, _ = self.greedy_sample(contexts)
            # baseline with bleu or CTR provider. temporal
            # baselines = self.reward_func_batch(greedy_pred, tgt)
            baselines = []
            for y, y_hat in zip(greedy_pred, tgt):
                baselines.append(self.reward_func(y, y_hat))
            baselines = torch.tensor(baselines).unsqueeze(
                1).expand_as(probs).cuda()
            rewards = rewards - baselines

        loss = -probs * rewards

        return loss, rewards, baselines, entropy

    def reward_func(self, y, y_hat, func='mlc'):
        """
        Define your own reward function. Predefined functions are mlc, bleu, rouge.
        :param y: target
        :param y_hat: prediction
        :param func: reward function
        :return: reward
        """
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
    #     reward = []
    #     for hypo, target in zip(ys, y_hats):
    #         self.bleu_scorer.reset()
    #         self.bleu_scorer.add(torch.IntTensor(hypo), torch.IntTensor(target))
    #         reward.append(self.bleu_scorer.score())
    #     # hypos = [self.tgt_vocab.convertToLabels(y, utils.EOS) for y in ys]
    #     # targets = [self.tgt_vocab.convertToLabels(y_hat, utils.EOS) for y_hat in y_hats]
    #     # reward = self.reward_provider.reward_fn_batched(hypos, targets)
    #     return reward

    def greedy_sample(self, contexts):
        """
        Greedy sampling
        :param contexts: source representations, [len, batch, size]
        :return: sampled ids.
        """
        bos = torch.ones(contexts.size(1)).long().fill_(utils.BOS).cuda()   # [batch]

        inputs, outputs = [bos], []
        for i in range(self.config.max_time_step):
            if self.config.positional:
                output, attn_weights = self.decoder(
                    inputs[i].unsqueeze(0), contexts, step=i)   # [len, batch, size]
            else:
                output, attn_weights, state = self.decoder(
                    inputs[i].unsqueeze(0), contexts, state, step=i)    # [len, batch, size]
            output = self.compute_score(output.transpose(0, 1)).squeeze(1)  # [batch, size]
            # torch.max(), specify dimension, output max and max indices.
            predicted = output.max(1)[1]
            inputs.append(predicted)
            outputs.append(predicted)
        sample_ids = torch.stack(outputs).t().tolist()  # [batch, len]

        return sample_ids

    def rl_sample(self, contexts):
        """
        Sampling for reinforcement learning
        :param contexts: source representations, [len, batch, size]
        :return: sampled ids, probability and entropy.
        """
        bos = torch.ones(contexts.size(1)).long().fill_(utils.BOS).cuda()   # [batch]

        inputs, outputs = [bos], []
        probs = []
        entropy = 0
        for i in range(self.config.max_time_step):
            if self.config.positional:
                output, attn_weights = self.decoder(
                    inputs[i].unsqueeze(0), contexts, step=i)   # [len, batch, size]
            else:
                output, attn_weights, state = self.decoder(
                    inputs[i].unsqueeze(0), contexts, state, step=i)    # [len, batch, size]
            output = self.compute_score(output.transpose(0, 1)).squeeze(1)  # [batch, vocab]
            entropy += (-F.softmax(output) * F.log_softmax(output)).sum(dim=1).data
            predicted = F.softmax(output).multinomial(1).squeeze()  # [batch]
            prob = F.log_softmax(output)[range(len(predicted)), predicted]
            probs += [prob]
            predicted = output.max(1)[1]
            inputs.append(predicted)
            outputs.append(predicted)
        sample_ids = torch.stack(outputs).t().tolist()
        probs = torch.stack(probs).squeeze()  # [max_tgt_len, batch]
        # probs = probs[:, reverse_indices]
        # entropy = entropy[reverse_indices]

        return sample_ids, probs, entropy

    def forward(self, src, src_len, dec, targets, teacher_ratio=1.0):
        """
        run transformer
        :param src: source input
        :param src_len: source length
        :param dec: decoder input
        :param targets: target
        :param teacher_ratio: ratio for teacher forcing
        :return: dictionary of loss, reward, etc., output scores
        """
        return_dict = {}
        src = src.t()   # [len, batch]
        dec = dec.t()   # [len, batch]
        targets = targets.t()   # [len, batch]

        if self.config.positional:
            contexts = self.encoder(src, src_len.tolist())  # [len, batch, size]
        else:
            contexts, state = self.encoder(src, src_len.tolist())   # [len, batch, size]

        models.transformer.init_state(self.decoder, src, contexts, self.decoder.num_layers)

        # printing for debugging
        #self.decoder.init_state(src, contexts)
        #print(src.device, id(self.decoder.init_state))
        #print(src.device, id(self.decoder))
        #print(src.device, self.decoder.state['cache'])
        if self.config.rl:
            rl_loss, rewards, baselines, entropy = self.compute_reward(
                contexts, targets.clone())
            # TODO entropy
            rl_loss = rl_loss
            return_dict['rl_loss'] = rl_loss.sum(dim=1).mean()
            return_dict['reward_mean'] = rewards[:, 0].mean()
            return_dict['greedy_mean'] = baselines[:, 0].mean()
            return_dict['sample_mean'] = return_dict['reward_mean'] + \
                return_dict['greedy_mean']
            return_dict['entropy'] = entropy.mean()

        if self.config.positional:
            outputs, attn_weights = self.decoder(dec, contexts) # [len, batch, size]
        else:
            outputs, attn_weights, state = self.decoder(dec, contexts, state)   # [len, batch, size]

        scores = self.compute_score(outputs.transpose(0, 1)).transpose(0, 1) # [len, batch, vocab]
        loss = self.compute_loss(scores, targets)
        return_dict['mle_loss'] = loss

        return return_dict, scores

    def sample(self, src, src_len):
        """
        Greedy sampling for inference
        :param src: source input
        :param src_len: source length
        :return: sampled ids and attention alignment
        """
        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, reverse_indices = torch.sort(indices)
        src = torch.index_select(src, dim=0, index=indices) # [batch, len]
        bos = torch.ones(src.size(0)).long().fill_(utils.BOS)   # [batch]
        if self.use_cuda:
            bos = bos.cuda()
        src = src.t()   # [len, batch]

        if self.config.positional:
            contexts = self.encoder(src, lengths.tolist())  # [len, batch, size]
        else:
            contexts, state = self.encoder(src, lengths.tolist())   # [len, batch, size]
        # self.decoder.init_state(src, contexts)
        models.transformer.init_state(self.decoder, src, contexts, self.decoder.num_layers)

        # sequential generation
        inputs, outputs, attn_matrix = [bos], [], []
        for i in range(self.config.max_time_step):
            if self.config.positional:
                output, attn_weights = self.decoder(
                    inputs[i].unsqueeze(0), contexts, step=i)   # [len, batch]
            else:
                output, attn_weights, state = self.decoder(
                    inputs[i].unsqueeze(0), contexts, state, step=i)    # [len, batch]
            output = self.compute_score(output.transpose(0, 1)).squeeze(1)  # [batch, size]
            predicted = output.max(1)[1]    # [batch]
            inputs.append(predicted)
            outputs.append(predicted)
            attn_matrix.append(attn_weights.squeeze(0)) #[batch, k_len]
        outputs = torch.stack(outputs)  # [len, batch]
        # select by the indices along the dimension of batch
        sample_ids = torch.index_select(
            outputs, dim=1, index=reverse_indices).t().tolist() # [batch, len]

        attn_matrix = torch.stack(attn_matrix)  # [len, batch, k_len]
        alignments = attn_matrix.max(2)[1]  # [len, batch]
        alignments = torch.index_select(
            alignments, dim=1, index=reverse_indices).t().tolist()  # [batch, len]

        return sample_ids, alignments

    def beam_sample(self, src, src_len, beam_size=1, eval_=False):
        """
        beam search
        :param src: source input
        :param src_len: source length
        :param beam_size: beam size
        :param eval_: evaluation or not
        :return: prediction, attention max score and attention weights
        """
        lengths, indices = torch.sort(src_len, dim=0, descending=True)  # [batch]
        _, ind = torch.sort(indices)
        src = torch.index_select(src, dim=0, index=indices) # [batch, len]
        src = src.t()   # [len, batch]
        batch_size = src.size(1)

        if self.config.positional:
            contexts = self.encoder(src, lengths.tolist())  # [len, batch, size]
        else:
            contexts, state = self.encoder(src, lengths.tolist())   # [len, batch, size]

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(batch_size, beam_size, -1)

        beam = [models.Beam(beam_size, n_best=1,
                            cuda=self.use_cuda, length_norm=self.config.length_norm)
                for __ in range(batch_size)]    # [batch, beam]

        contexts = tile(contexts, beam_size, 1) # [len, batch*beam, size]
        src = tile(src, beam_size, 1)   # [len, batch*beam]

        if not self.config.positional:
            h = tile(state[0], beam_size, 0)
            c = tile(state[1], beam_size, 0)
            state = (h, c)  # [len, batch*beam, size]

        # self.decoder.init_state(src, contexts)
        models.transformer.init_state(self.decoder, src, contexts, self.decoder.num_layers)

        # sequential generation
        for i in range(self.config.max_time_step):
            # finish beam search
            if all((b.done() for b in beam)):
                break

            inp = torch.stack([b.getCurrentState() for b in beam])
            inp = inp.view(1, -1)   # [1, batch*beam]

            if self.config.positional:
                output, attn = self.decoder(inp, contexts, step=i)  # [len, batch*beam, size]
                state = None
            else:
                output, attn, state = self.decoder(
                    inp, contexts, state, step=i)   # [len, batch*beam, size]
            output = self.compute_score(output.transpose(0, 1)).squeeze(1)  # [batch*beam, size]

            output = unbottle(self.log_softmax(output)) # [batch, beam, size]
            attn = unbottle(attn.squeeze(0))    # [batch, beam, k_len]

            select_indices_array = []
            # scan beams in a batch
            for j, b in enumerate(beam):
                # update each beam
                b.advance(output[j, :], attn[j, :]) # batch index
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
