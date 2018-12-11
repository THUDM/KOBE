import torch
import torch.nn as nn
import utils
import models
import random


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
                 src_padding_idx=0, tgt_padding_idx=0, label_smoothing=0):
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
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=utils.PAD, reduction='none')
        if config.use_cuda:
            self.criterion.cuda()

    def compute_loss(self, scores, targets):
        scores = scores.view(-1, scores.size(2))
        loss = self.criterion(scores, targets.contiguous().view(-1))
        return loss

    def forward(self, src, src_len, dec, targets, teacher_ratio=1.0):
        src = src.t()
        dec = dec.t()
        targets = targets.t()
        teacher = random.random() < teacher_ratio

        contexts, state = self.encoder(src, src_len.tolist())

        if self.decoder.attention is not None:
            self.decoder.attention.init_context(context=contexts)
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
        return loss, outputs

    def sample(self, src, src_len):

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, reverse_indices = torch.sort(indices)
        src = torch.index_select(src, dim=0, index=indices)
        bos = torch.ones(src.size(0)).long().fill_(utils.BOS)
        src = src.t()

        if self.use_cuda:
            bos = bos.cuda()

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
            outputs, dim=1, index=reverse_indices).t()

        if self.decoder.attention is not None:
            attn_matrix = torch.stack(attn_matrix)
            alignments = attn_matrix.max(2)[1]
            alignments = torch.index_select(
                alignments, dim=1, index=reverse_indices).t()
        else:
            alignments = None

        return sample_ids.tolist(), alignments

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

        # for j in ind.data:
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
