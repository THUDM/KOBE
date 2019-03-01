import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import models
import math
import numpy as np
import utils
from torch.distributions.bernoulli import Bernoulli
import random


def add_unk(input_token_id, p):
    # random.random() gives you a value between 0 and 1
    # to avoid switching your padding to 0 we add 'input_token_id > 1'
    if random.random() < p and input_token_id > 1:
        return 0
    else:
        return input_token_id


class rnn_encoder(nn.Module):

    def __init__(self, config, embedding=None, padding_idx=0):
        super(rnn_encoder, self).__init__()

        self.embedding = embedding if embedding is not None else nn.Embedding(
            config.src_vocab_size, config.emb_size, padding_idx=0)
        self.hidden_size = config.hidden_size
        self.config = config
        self.padding_idx = padding_idx
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        if config.swish:
            self.sw1 = nn.Sequential(
                nn.Conv1d(config.hidden_size, config.hidden_size,
                          kernel_size=1, padding=0), nn.BatchNorm1d(config.hidden_size), nn.ReLU())
            self.sw3 = nn.Sequential(
                nn.Conv1d(config.hidden_size, config.hidden_size,
                          kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm1d(config.hidden_size),
                nn.Conv1d(config.hidden_size, config.hidden_size,
                          kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(config.hidden_size))
            self.sw33 = nn.Sequential(
                nn.Conv1d(config.hidden_size, config.hidden_size,
                          kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm1d(config.hidden_size),
                nn.Conv1d(config.hidden_size, config.hidden_size,
                          kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(config.hidden_size),
                nn.Conv1d(config.hidden_size, config.hidden_size,
                          kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(config.hidden_size))
            self.linear = nn.Sequential(nn.Linear(
                2*config.hidden_size, 2*config.hidden_size), nn.GLU(), nn.Dropout(config.dropout))
            self.filter_linear = nn.Linear(
                3*config.hidden_size, config.hidden_size)
            self.tanh = nn.Tanh()
            self.sigmoid = nn.Sigmoid()

        if config.selfatt:
            self.self_attn = models.Multihead_Attention(
                config.hidden_size, head_count=config.heads, dropout=config.dropout)

        if config.cell == 'gru':
            self.rnn = nn.GRU(input_size=config.emb_size, hidden_size=config.hidden_size,
                              num_layers=config.enc_num_layers, dropout=config.dropout,
                              bidirectional=config.bidirectional)
        else:
            self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.hidden_size,
                               num_layers=config.enc_num_layers, dropout=config.dropout,
                               bidirectional=config.bidirectional)
            self.dropout = nn.Dropout(config.dropout)
            self.emb_drop = nn.Dropout(config.emb_dropout)

    def forward(self, inputs, lengths):
        #probs = torch.empty(inputs.size(), device='cuda').uniform_(0, 1)
        # inputs = torch.where(probs < self.config.emb_dropout, inputs,
                             # torch.zeros_like(inputs))
        embs = pack(self.emb_drop(self.embedding(inputs)), lengths)
        #mask = Bernoulli(1 - self.config.emb_dropout).sample((embs.shape[0],))
        #embs = pack(embs.transpose(0, 1)[:, mask==1].transpose(0, 1), lengths)
        self.rnn.flatten_parameters()
        outputs, state = self.rnn(embs)
        outputs = unpack(outputs)[0]
        if self.config.bidirectional:
            if self.config.swish:
                outputs = self.linear(outputs)
            else:
                outputs = outputs[:, :, :self.config.hidden_size] + \
                    outputs[:, :, self.config.hidden_size:]
                outputs = self.dropout(outputs)
        if self.config.swish:
            outputs = outputs.transpose(0, 1).transpose(1, 2)
            conv1 = self.sw1(outputs)
            conv3 = self.sw3(outputs)
            conv33 = self.sw33(outputs)
            conv = torch.cat((conv1, conv3, conv33), 1)
            conv = self.filter_linear(conv.transpose(1, 2))
            if self.config.selfatt:
                outputs = outputs.transpose(1, 2).transpose(0, 1)
            else:
                gate = self.sigmoid(conv)
                outputs = outputs * gate.transpose(1, 2)
                outputs = outputs.transpose(1, 2).transpose(0, 1)

        if self.config.selfatt:
            src_words = inputs.transpose(0, 1)
            src_batch, src_len = src_words.size()
            padding_idx = self.padding_idx
            mask = src_words.data.eq(padding_idx).unsqueeze(1) \
                .expand(src_batch, src_len, src_len)
            if self.config.swish:
                context, _ = self.self_attn(conv, conv, conv,
                                            mask=mask)
                gate = self.sigmoid(context)
                outputs = outputs * gate.transpose(0, 1)
            else:
                outputs = outputs.transpose(0, 1)
                context, _ = self.self_attn(outputs, outputs, outputs,
                                            mask=mask)
                if self.config.gate:
                    outputs = context.transpose(0, 1) * outputs.transpose(0, 1)
                else:
                    outputs = context.transpose(0, 1) + outputs.transpose(0, 1)
                # outputs = self.layer_norm(outputs)

        if self.config.cell == 'gru':
            state = state[:self.config.dec_num_layers]
        else:
            state = (state[0][::2], state[1][::2])

        return outputs, state


class rnn_decoder(nn.Module):

    def __init__(self, config, embedding=None, use_attention=True, padding_idx=0):
        super(rnn_decoder, self).__init__()
        self.embedding = embedding if embedding is not None else nn.Embedding(
            config.tgt_vocab_size, config.emb_size, padding_idx=0)

        input_size = config.emb_size

        if config.cell == 'gru':
            self.rnn = StackedGRU(input_size=input_size, hidden_size=config.hidden_size,
                                  num_layers=config.dec_num_layers, dropout=config.dropout)
        else:
            self.rnn = StackedLSTM(input_size=input_size, hidden_size=config.hidden_size,
                                   num_layers=config.dec_num_layers, dropout=config.dropout)

        self.linear = nn.Linear(config.hidden_size, config.tgt_vocab_size)
        self.linear_ = nn.Linear(config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()

        if not use_attention or config.attention == 'None':
            self.attention = None
        elif config.attention == 'bahdanau':
            self.attention = models.bahdanau_attention(
                config.hidden_size, config.emb_size, config.pool_size)
        elif config.attention == 'luong':
            self.attention = models.luong_attention(config.hidden_size)
        elif config.attention == 'luong_gate':
            self.attention = models.luong_gate_attention(
                config.hidden_size, config.emb_size, prob=config.dropout)

        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.emb_drop = nn.Dropout(config.emb_dropout)
        self.config = config

    def forward(self, input, state):
        #probs = torch.empty(input.size(), device='cuda').uniform_(0, 1)
        # input = torch.where(probs < self.config.emb_dropout,
                            # input, torch.zeros_like(input))
        embs = self.emb_drop(self.embedding(input))
        output, state = self.rnn(embs, state)
        if self.attention is not None:
            if self.config.attention == 'luong_gate':
                output, attn_weights = self.attention(
                    output, Bernoulli=self.config.Bernoulli)
            else:
                output, attn_weights = self.attention(output)
        else:
            attn_weights = None

        output = self.compute_score(output)

        return output, state, attn_weights

    def compute_score(self, hiddens):
        scores = self.linear(hiddens)
        return scores


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1
