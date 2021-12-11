import math

import models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from models import rnn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

MAX_SIZE = 5000


class PositionalEncoding(nn.Module):
    """positional encoding"""

    def __init__(self, dropout, dim, max_len=5000):
        """
        initialization of required variables and functions
        :param dropout: dropout probability
        :param dim: hidden size
        :param max_len: maximum length
        """
        # positional encoding initialization
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        # term to divide
        div_term = torch.exp(
            (torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        )
        # sinusoidal positional encoding
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """
        create positional encoding
        :param emb: word embedding
        :param step: step for decoding in inference
        :return: positional encoding representation
        """
        # division of size
        emb = emb * math.sqrt(self.dim)
        if step is None:
            # residual connection
            emb = emb + self.pe[: emb.size(0)]  # [len, batch, size]
        else:
            # step for inference
            emb = emb + self.pe[step]  # [len, batch, size]
        emb = self.dropout(emb)
        return emb


class PositionwiseFeedForward(nn.Module):
    """Point-wise Feed-Forward NN, FFN, in fact 1-d convolution"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        initialization of required functions
        :param d_model: model size
        :param d_ff: intermediate size
        :param dropout: dropout probability
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        run FFN
        :param x: input
        :return: output
        """
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        # with residual connection
        return output + x


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer"""

    def __init__(self, config):
        """
        initialization of required variables and functions
        :param config: configuration
        """
        super(TransformerEncoderLayer, self).__init__()
        self.config = config
        # self attention
        self.self_attn = models.Multihead_Attention(
            model_dim=config.hidden_size,
            head_count=config.heads,
            dropout=config.dropout,
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model=config.hidden_size, d_ff=config.d_ff, dropout=config.dropout
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(config.dropout)

        # Convolutional Attention Temperature for self attention distribution, waiting to be deprecated
        if config.convolutional:
            self.cnn_tau = nn.Sequential(
                nn.Conv1d(
                    config.hidden_size,
                    config.heads,
                    kernel_size=3,
                    padding=1,
                    groups=config.heads,
                ),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
            self.ln_tau = nn.LayerNorm(config.heads, eps=1e-6)
            self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, mask):
        """
        run transformer encoder layer
        :param inputs: inputs
        :param mask: mask
        :return: output
        """
        # self attention
        input_norm = self.layer_norm(inputs)  # [batch, len, size]
        if self.config.convolutional:
            # multiple transpose operation for the size adaptation to the functions
            # generate tau for the control of attention distribution
            tau = self.cnn_tau(input_norm.transpose(1, 2)).transpose(
                1, 2
            )  # [batch, len, 1]
            tau = self.sigmoid(self.ln_tau(tau))  # [batch, len, 1]
            context, _ = self.self_attn(
                input_norm, input_norm, input_norm, mask=mask, tau=tau
            )  # [batch, len, size]
        else:
            context, _ = self.self_attn(
                input_norm, input_norm, input_norm, mask=mask
            )  # [batch, len, size]
        out = self.dropout(context) + inputs  # [batch, len, size]

        # FFN
        return self.feed_forward(out)  # [batch, len, size]


class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(
            torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5))
        )

    def forward(self, input, memory, mask=None):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(
            input * self.dot_scale, memory.permute(0, 2, 1).contiguous()
        )
        att = input_dot + memory_dot + cross_dot
        if mask is not None:
            att = att - 1e30 * (1 - mask[:, None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat(
            [input, output_one, input * output_one, output_two * output_one], dim=-1
        )


class TransformerEncoder(nn.Module):
    """Transformer encoder"""

    def __init__(self, config, padding_idx=0):
        """
        initialization of required variables and functions
        :param config: configuration
        :param padding_idx: index for padding in the dictionary
        """
        super(TransformerEncoder, self).__init__()

        self.config = config
        self.num_layers = config.enc_num_layers

        # HACK: 512 for word embeddings, 512 for condition embeddings
        self.embedding = nn.Embedding(
            config.src_vocab_size, config.emb_size, padding_idx=padding_idx
        )
        # positional encoding
        if config.positional:
            self.position_embedding = PositionalEncoding(
                config.dropout, config.emb_size
            )
        else:
            # RNN for positional information, waiting to be deprecated
            self.rnn = nn.LSTM(
                input_size=config.emb_size,
                hidden_size=config.hidden_size,
                num_layers=1,
                dropout=0,
                bidirectional=config.bidirectional,
            )

        # transformer
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.enc_num_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.padding_idx = padding_idx
        self.condition_context_attn = BiAttention(config.hidden_size, config.dropout)
        self.bi_attn_transform = nn.Linear(config.hidden_size * 4, config.hidden_size)

    def forward(self, src, lengths=None, is_knowledge=False):
        """
        run transformer encoder
        :param src: source input
        :param lengths: sorted lengths
        :return: output and state (if with rnn)
        """
        if self.config.conditioned and not is_knowledge:
            # HACK: recover the original sentence without the condition
            conditions_1 = src[[length - 1 for length in lengths], range(src.shape[1])]
            conditions_2 = src[[length - 2 for length in lengths], range(src.shape[1])]
            src[[length - 1 for length in lengths], range(src.shape[1])] = utils.PAD
            src[[length - 2 for length in lengths], range(src.shape[1])] = utils.PAD
            lengths = [length - 2 for length in lengths]
            assert all([length > 0 for length in lengths])
            # print(conditions.shape) # batch_size
            # print(src.shape) # max_len X batch_size
            conditions_1 = conditions_1.unsqueeze(0)  # 1 X batch_size
            conditions_2 = conditions_2.unsqueeze(0)  # 1 X batch_size
        embed = self.embedding(src)

        # RNN for positional information
        if self.config.positional:
            emb = self.position_embedding(embed)  # [len, batch, size]
        else:
            emb, state = self.rnn(pack(embed, lengths))
            emb = unpack(emb)[0]  # [len, batch, 2*size]
            emb = (
                emb[:, :, : self.config.hidden_size]
                + emb[:, :, self.config.hidden_size :]
            )  # [len, batch, size]
            emb = emb + embed  # [len, batch, size]
            state = (state[0][0], state[1][0])  # LSTM states

        if self.config.conditioned and not is_knowledge:
            assert self.config.positional
            conditions_1_embed = self.embedding(conditions_1)
            conditions_1_embed = conditions_1_embed.expand_as(embed)
            conditions_2_embed = self.embedding(conditions_2)
            conditions_2_embed = conditions_2_embed.expand_as(embed)
            # Concat
            # emb = torch.cat([emb, conditions_embed], dim=-1)
            # emb = self.embed_transform(emb)
            # emb = torch.cat([emb, conditions_1_embed + conditions_2_embed], dim=-1)
            # emb = self.embed_transform(emb)
            # Add
            # emb = emb + conditions_embed
            emb = emb + conditions_1_embed + conditions_2_embed
            # Remove condition
            # emb = emb

        out = emb.transpose(0, 1).contiguous()  # [batch, len, size]
        src_words = src.transpose(0, 1)  # [batch, len]
        src_batch, src_len = src_words.size()
        padding_idx = self.padding_idx
        mask = (
            src_words.data.eq(padding_idx)
            .unsqueeze(1)
            .expand(src_batch, src_len, src_len)
        )  # [batch, len, len]

        for i in range(self.num_layers):
            out = self.transformer[i](out, mask)
        out = self.layer_norm(out)  # [batch, len, size]

        assert self.config.positional
        if self.config.positional:
            # out = self.condition_context_attn(out, conditions_embed)
            # out = self.bi_attn_control_exp(out)
            return out.transpose(0, 1)
        else:
            return out.transpose(0, 1), state  # [len, batch, size]


# Decoder
class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer"""

    def __init__(self, config):
        """
        initialization for required variables and functions
        :param config: configuration
        """
        super(TransformerDecoderLayer, self).__init__()
        self.config = config
        # self attention
        self.self_attn = models.Multihead_Attention(
            model_dim=config.hidden_size,
            head_count=config.heads,
            dropout=config.dropout,
        )

        self.context_attn = models.Multihead_Attention(
            model_dim=config.hidden_size,
            head_count=config.heads,
            dropout=config.dropout,
        )
        self.feed_forward = PositionwiseFeedForward(
            config.hidden_size, config.d_ff, config.dropout
        )
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.dropout = config.dropout
        self.drop = nn.Dropout(config.dropout)
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer("mask", mask)

        # Add convolutional temperature for attention distribution, to be deprecated
        if config.convolutional:
            self.self_lin = nn.Sequential(
                nn.Linear(config.hidden_size, config.heads), nn.ReLU(), nn.Dropout()
            )
            self.self_ln = nn.LayerNorm(config.heads, eps=1e-6)
            self.self_sigmoid = nn.Sigmoid()
            self.ctxt_lin = nn.Sequential(
                nn.Linear(config.hidden_size, config.heads),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
            self.ctxt_ln = nn.LayerNorm(config.heads, eps=1e-6)
            self.ctxt_sigmoid = nn.Sigmoid()

    def forward(
        self,
        inputs,
        memory_bank,
        src_pad_mask,
        tgt_pad_mask,
        layer_cache=None,
        step=None,
    ):
        """
        run transformer decoder layer
        :param inputs: inputs
        :param memory_bank: source representations
        :param src_pad_mask: source padding mask
        :param tgt_pad_mask: target padding mask
        :param layer_cache: layer cache for decoding in inference stage
        :param step: step for decoding in inference stage
        :return: output, attention weights and input norm
        """
        dec_mask = torch.gt(
            tgt_pad_mask + self.mask[:, : tgt_pad_mask.size(1), : tgt_pad_mask.size(1)],
            0,
        )

        # self attention
        input_norm = self.layer_norm_1(inputs)
        if self.config.convolutional:
            # generate tau for the control of attention distribution, to be deprecated
            tau_self = self.self_lin(input_norm)  # N*L*C
            tau_self = self.self_sigmoid(self.self_ln(tau_self))
            query, attn = self.self_attn(
                input_norm,
                input_norm,
                input_norm,
                mask=dec_mask,
                layer_cache=layer_cache,
                type="self",
                tau=tau_self,
            )
        else:
            query, attn = self.self_attn(
                input_norm,
                input_norm,
                input_norm,
                mask=dec_mask,
                layer_cache=layer_cache,
                type="self",
            )  # [batch, q_len, size]
        # residual connection
        query = self.drop(query) + inputs  # [batch, q_len, size]

        # context attention
        query_norm = self.layer_norm_2(query)
        if self.config.convolutional:
            # generate tau for the control of attention distribution, to be deprecated
            tau_ctxt = self.ctxt_lin(query_norm)  # [batch, len, size]
            tau_ctxt = self.ctxt_sigmoid(self.ctxt_ln(tau_ctxt))
            mid, attn = self.context_attn(
                memory_bank,
                memory_bank,
                query_norm,
                mask=src_pad_mask,
                layer_cache=layer_cache,
                type="context",
                tau=tau_ctxt,
            )
        else:
            mid, attn = self.context_attn(
                memory_bank,
                memory_bank,
                query_norm,
                mask=src_pad_mask,
                layer_cache=layer_cache,
                type="context",
                Bernoulli=self.config.Bernoulli,
            )  # [batch, q_len, size]

        # FFN
        output = self.feed_forward(self.drop(mid) + query)  # [batch, q_len, size]

        return output, attn, input_norm

    def _get_attn_subsequent_mask(self, size):
        """
        get mask for target
        :param size: max size
        :return: target mask
        """
        attn_shape = (1, size, size)  # [1, size, size]
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoder(nn.Module):
    """Transformer decoder"""

    def __init__(self, config, tgt_embedding=None, padding_idx=0):
        """
        initialization for required variables and functions
        :param config: configuration
        :param tgt_embedding: target embedding
        :param padding_idx: padding index
        """
        super(TransformerDecoder, self).__init__()
        self.config = config
        self.num_layers = config.dec_num_layers
        if tgt_embedding:
            self.embedding = tgt_embedding
        else:
            self.embedding = nn.Embedding(
                config.tgt_vocab_size, config.emb_size, padding_idx=padding_idx
            )
        if config.positional:
            self.position_embedding = PositionalEncoding(
                config.dropout, config.emb_size
            )
        else:
            self.rnn = nn.LSTMCell(config.emb_size, config.hidden_size)

        self.padding_idx = padding_idx
        # state to store elements, including source and layer cache
        self.state = {}
        # transformer decoder
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(config) for _ in range(config.dec_num_layers)]
        )

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def init_state(self, src, memory_bank):
        """
        initialization for state, to be deprecated. Use init_state outside the class instead.
        :param src: source input
        :param memory_bank: source representations
        :return: none
        """
        self.state["src"] = src
        # initialization of layer cache
        self._init_cache(memory_bank, self.num_layers)

    def map_state(self, fn):
        """
        state mapping
        :param fn: function
        :return: none
        """

        def _recursive_map(struct, batch_dim=0):
            """
            recursive mapping
            :param struct: object for mapping
            :param batch_dim: batch dimension
            :return: none
            """
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        # self.state["src"] = fn(self.state["src"], 1)
        # layer cache mapping
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def forward(self, tgt, memory_bank, state=None, step=None):
        """
        run transformer decoder
        :param tgt: target input
        :param memory_bank: source representations
        :param state: state
        :param step: step for inference
        :return: output, attention weights and state
        """
        src = self.state["src"]
        src_words = src.transpose(0, 1)  # [batch, len]
        tgt_words = tgt.transpose(0, 1)  # [batch, len]
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        if self.config.positional:
            emb = self.embedding(tgt)  # [len, batch, size]
            emb = self.position_embedding(emb, step=step)
        else:
            # for rnn, to be deprecated
            emb = []
            for inp in tgt.split(1):
                inp = self.embedding(inp.squeeze(0))
                h_, c = self.rnn(inp, state)
                h = inp + h_
                emb.append(h)
            emb = torch.stack(emb)
            state = (h_, c)

        output = emb.transpose(0, 1).contiguous()  # [batch, len, size]
        src_memory_bank = memory_bank.transpose(0, 1)  # [batch, len, size]

        padding_idx = self.padding_idx
        # source padding mask
        src_pad_mask = (
            src_words.data.eq(padding_idx)
            .unsqueeze(1)
            .expand(src_batch, tgt_len, src_len)
        )  # [batch, tgt_len, src_len]
        # target padding mask
        tgt_pad_mask = (
            tgt_words.data.eq(padding_idx)
            .unsqueeze(1)
            .expand(tgt_batch, tgt_len, tgt_len)
        )  # [batch, tgt_len, tgt_len]

        # run transformer decoder layers
        for i in range(self.num_layers):
            output, attn, all_input = self.transformer_layers[i](
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=self.state["cache"]["layer_{}".format(i)],
                step=step,
            )

        output = self.layer_norm(output)  # [batch, len, size]

        dec_outs = output.transpose(0, 1).contiguous()  # [len, batch, size]
        attn = attn.transpose(0, 1).contiguous()  # [q_len, batch, k_len]

        if self.config.positional:
            return dec_outs, attn
        else:
            return dec_outs, attn, state

    def _init_cache(self, memory_bank, num_layers):
        """
        layer cache initialization
        :param memory_bank: source representations
        :param num_layers: number of layers
        :return: none
        """
        self.state["cache"] = {}
        device = str(memory_bank.device)
        # print(device)

        memory_keys = "memory_keys_" + device
        memory_values = "memory_values_" + device
        self_keys = "self_keys_" + device
        # print(self_keys)
        self_values = "self_values_" + device

        # build layer cache for each layer
        for l in range(num_layers):
            layer_cache = {
                memory_keys: None,
                memory_values: None,
                self_keys: None,
                self_values: None,
            }
            # store in the cache in state
            self.state["cache"]["layer_{}".format(l)] = layer_cache


def init_state(self, src, memory_bank, num_layers):
    """
    state initialization, to replace the one in the transformer decoder
    :param self: self
    :param src: source input
    :param memory_bank: source representations
    :param num_layers: number of layers
    :return: none
    """
    self.state = {}
    self.state["src"] = src
    self.state["cache"] = {}

    # device for multi-gpus
    device = str(memory_bank.device)
    # print(device)

    memory_keys = "memory_keys_" + device
    memory_values = "memory_values_" + device
    self_keys = "self_keys_" + device
    # print(self_keys)
    self_values = "self_values_" + device

    # build layer cache for each layer
    for l in range(num_layers):
        layer_cache = {
            memory_keys: None,
            memory_values: None,
            self_keys: None,
            self_values: None,
        }
        # store in the cache in state
        self.state["cache"]["layer_{}".format(l)] = layer_cache
