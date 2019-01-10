import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import math
import models
from models import rnn
import numpy as np


MAX_SIZE = 5000


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class TransformerEncoderLayer(nn.Module):

    def __init__(self, config):
        super(TransformerEncoderLayer, self).__init__()

        self.config = config

        self.self_attn = models.Multihead_Attention(
            model_dim=config.hidden_size, head_count=config.heads, dropout=config.dropout)
        self.feed_forward = PositionwiseFeedForward(
            d_model=config.hidden_size, d_ff=config.d_ff, dropout=config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(config.dropout)

        # Convolutional Attention Temperature for self attention distribution
        if config.convolutional:
            self.cnn_tau = nn.Sequential(
                nn.Conv1d(config.hidden_size, config.heads,
                          kernel_size=3, padding=1, groups=config.heads),
                nn.ReLU(), nn.Dropout(config.dropout))
            self.ln_tau = nn.LayerNorm(config.heads, eps=1e-6)
            self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, mask):
        input_norm = self.layer_norm(inputs)  # N * L * C
        if self.config.convolutional:
            # multiple transpose operation for the size adaptation to the functions
            # generate tau for the control of attention distribution
            tau = self.cnn_tau(input_norm.transpose(1, 2)).transpose(1, 2)
            tau = self.sigmoid(self.ln_tau(tau))
            context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                        mask=mask, tau=tau)
        else:
            context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                        mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):

    def __init__(self, config, padding_idx=0):
        super(TransformerEncoder, self).__init__()

        self.config = config
        self.num_layers = config.enc_num_layers

        self.embedding = nn.Embedding(config.src_vocab_size, config.emb_size,
                                      padding_idx=padding_idx)
        if config.positional:
            self.position_embedding = PositionalEncoding(
                config.dropout, config.emb_size)
        else:
            # RNN for positional information
            self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.hidden_size,
                               num_layers=1, dropout=0, bidirectional=config.bidirectional)

        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(config)
             for _ in range(config.enc_num_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.padding_idx = padding_idx

    def forward(self, src, lengths=None):

        embed = self.embedding(src)

        # RNN for positional information
        if self.config.positional:
            emb = self.position_embedding(embed)
        else:
            emb, state = self.rnn(pack(embed, lengths))
            emb = unpack(emb)[0]
            emb = emb[:, :, :self.config.hidden_size] + \
                emb[:, :, self.config.hidden_size:]
            emb = emb + embed
            state = (state[0][0], state[1][0])

        out = emb.transpose(0, 1).contiguous()
        src_words = src.transpose(0, 1)
        src_batch, src_len = src_words.size()
        padding_idx = self.padding_idx
        mask = src_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(src_batch, src_len, src_len)

        for i in range(self.num_layers):
            out = self.transformer[i](out, mask)
        out = self.layer_norm(out)

        if self.config.positional:
            return out.transpose(0, 1)
        else:
            return out.transpose(0, 1), state


# Decoder
class TransformerDecoderLayer(nn.Module):

    def __init__(self, config):
        super(TransformerDecoderLayer, self).__init__()

        self.config = config

        self.self_attn = models.Multihead_Attention(
            model_dim=config.hidden_size, head_count=config.heads, dropout=config.dropout)

        self.context_attn = models.Multihead_Attention(
            model_dim=config.hidden_size, head_count=config.heads, dropout=config.dropout)
        self.feed_forward = PositionwiseFeedForward(
            config.hidden_size, config.d_ff, config.dropout)
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.dropout = config.dropout
        self.drop = nn.Dropout(config.dropout)
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)

        # Add convolutional temperature for attention distribution
        if config.convolutional:
            self.self_lin = nn.Sequential(
                nn.Linear(config.hidden_size, config.heads),
                nn.ReLU(), nn.Dropout())
            self.self_ln = nn.LayerNorm(config.heads, eps=1e-6)
            self.self_sigmoid = nn.Sigmoid()
            self.ctxt_lin = nn.Sequential(
                nn.Linear(config.hidden_size, config.heads),
                nn.ReLU(), nn.Dropout(config.dropout))
            self.ctxt_ln = nn.LayerNorm(config.heads, eps=1e-6)
            self.ctxt_sigmoid = nn.Sigmoid()

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                layer_cache=None, step=None):
        dec_mask = torch.gt(tgt_pad_mask +
                            self.mask[:, :tgt_pad_mask.size(1), :tgt_pad_mask.size(1)], 0)

        input_norm = self.layer_norm_1(inputs)
        if self.config.convolutional:
            # generate tau for the control of attention distribution
            tau_self = self.self_lin(input_norm) # N*L*C
            tau_self = self.self_sigmoid(self.self_ln(tau_self))
            query, attn = self.self_attn(input_norm, input_norm, input_norm,
                                         mask=dec_mask,
                                         layer_cache=layer_cache,
                                         type="self",
                                         tau=tau_self)
        else:
            query, attn = self.self_attn(input_norm, input_norm, input_norm,
                                         mask=dec_mask,
                                         layer_cache=layer_cache,
                                         type="self")
        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        if self.config.convolutional:
            # generate tau for the control of attention distribution
            tau_ctxt = self.ctxt_lin(query_norm) # N*L*C
            tau_ctxt = self.ctxt_sigmoid(self.ctxt_ln(tau_ctxt))
            mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                          mask=src_pad_mask,
                                          layer_cache=layer_cache,
                                          type="context",
                                          tau=tau_ctxt)
        else:
            mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                          mask=src_pad_mask,
                                          layer_cache=layer_cache,
                                          type="context")

        output = self.feed_forward(self.drop(mid) + query)

        return output, attn, input_norm

    def _get_attn_subsequent_mask(self, size):
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoder(nn.Module):

    def __init__(self, config, tgt_embedding=None, padding_idx=0):
        super(TransformerDecoder, self).__init__()

        self.config = config

        self.num_layers = config.dec_num_layers
        if tgt_embedding:
            self.embedding = tgt_embedding
        else:
            self.embedding = nn.Embedding(config.tgt_vocab_size, config.emb_size,
                                          padding_idx=padding_idx)
        if config.positional:
            self.position_embedding = PositionalEncoding(
                config.dropout, config.emb_size)
        else:
            self.rnn = nn.LSTMCell(config.emb_size, config.hidden_size)

        self.padding_idx = padding_idx

        self.state = {}

        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(config)
             for _ in range(config.dec_num_layers)])

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def init_state(self, src, memory_bank):
        self.state["src"] = src
        self._init_cache(memory_bank, self.num_layers)

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)
        # self.state["src"] = fn(self.state["src"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def forward(self, tgt, memory_bank, state=None, step=None):
        src = self.state["src"]
        src_words = src.transpose(0, 1)
        tgt_words = tgt.transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        dec_outs = []

        if self.config.positional:
            emb = self.embedding(tgt)
            emb = self.position_embedding(emb, step=step)
        else:
            emb = []
            for inp in tgt.split(1):
                inp = self.embedding(inp.squeeze(0))
                h_, c = self.rnn(inp, state)
                h = inp + h_
                emb.append(h)
            emb = torch.stack(emb)
            state = (h_, c)

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1)

        padding_idx = self.padding_idx
        src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(src_batch, tgt_len, src_len)
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)

        for i in range(self.num_layers):
            output, attn, all_input = self.transformer_layers[i](
                output, src_memory_bank,
                src_pad_mask, tgt_pad_mask,
                layer_cache=self.state["cache"]["layer_{}".format(i)],
                step=step)

        output = self.layer_norm(output)

        dec_outs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        if self.config.positional:
            return dec_outs, attn
        else:
            return dec_outs, attn, state

    def _init_cache(self, memory_bank, num_layers):
        self.state["cache"] = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for l in range(num_layers):
            layer_cache = {
                "memory_keys": None,
                "memory_values": None,
                "self_keys": None,
                "self_values": None
            }
            self.state["cache"]["layer_{}".format(l)] = layer_cache
