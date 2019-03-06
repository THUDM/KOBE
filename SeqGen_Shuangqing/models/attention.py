import math

import torch
import torch.nn as nn


class luong_attention(nn.Module):
    """
    Luong's global attention
    """
    def __init__(self, hidden_size):
        super(luong_attention, self).__init__()
        self.hidden_size = hidden_size
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.Tanh())
        self.softmax = nn.Softmax(dim=-1)

    def init_context(self, context):
        self.context = context.transpose(0, 1)

    def forward(self, h):
        gamma_h = self.linear_in(h).unsqueeze(2)    # [batch, size, 1]
        weights = torch.bmm(self.context, gamma_h).squeeze(2)   # [batch, len]
        weights = self.softmax(weights)   # [batch, len]
        c_t = torch.bmm(weights.unsqueeze(
            1), self.context).squeeze(1)  # [batch, size]
        output = self.linear_out(torch.cat([c_t, h], 1))

        return output, weights


class luong_gate_attention(nn.Module):
    """
    Junyang's attention, with Selu function and some setting for multi-label classification.
    """

    def __init__(self, hidden_size, emb_size, prob=0.1):
        super(luong_gate_attention, self).__init__()
        self.hidden_size, self.emb_size = hidden_size, emb_size
        self.linear_enc = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob),
                                        nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.linear_in = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob),
                                       nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.linear_out = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob),
                                        nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=prob)
        self.sigmoid = nn.Sigmoid()

    def init_context(self, context):
        self.context = context.transpose(0, 1)

    def forward(self, h, Bernoulli=False):
        gamma_h = self.linear_in(h).unsqueeze(2)    # [batch, size, 1]
        weights = torch.bmm(self.context, gamma_h).squeeze(2)   # [batch, len]
        # For multi-label classification
        # implementation of Inverse Attention
        if Bernoulli:
            weights = self.sigmoid(weights)  # [batch, len]
        else:
            weights = self.softmax(weights)  # [batch, len]
        c_t = torch.bmm(weights.unsqueeze(1), self.context).squeeze(1)  # [batch, size]
        output = self.linear_out(torch.cat([h, c_t], 1))

        return output, weights


class bahdanau_attention(nn.Module):
    """
    Bahdanau's attention
    """

    def __init__(self, hidden_size, emb_size, pool_size=0):
        super(bahdanau_attention, self).__init__()
        self.linear_encoder = nn.Linear(hidden_size, hidden_size)
        self.linear_decoder = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, 1)
        self.linear_r = nn.Linear(hidden_size * 2 + emb_size, hidden_size * 2)
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def init_context(self, context):
        self.context = context.transpose(0, 1)

    def forward(self, h, x):
        gamma_encoder = self.linear_encoder(
            self.context)   # [batch, len, size]
        gamma_decoder = self.linear_decoder(
            h).unsqueeze(1)    # [batch, 1, size]
        weights = self.linear_v(
            self.tanh(gamma_encoder + gamma_decoder)).squeeze(2)   # [batch, len]
        weights = self.softmax(weights)   # [batch, len]
        c_t = torch.bmm(weights.unsqueeze(
            1), self.context).squeeze(1)  # [batch, size]
        r_t = self.linear_r(torch.cat([c_t, h, x], dim=1))
        output = r_t.view(-1, self.hidden_size, 2).max(2)[0]

        return output, weights


class maxout(nn.Module):
    """
    maxout network
    """

    def __init__(self, in_feature, out_feature, pool_size):
        super(maxout, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.pool_size = pool_size
        self.linear = nn.Linear(in_feature, out_feature * pool_size)

    def forward(self, x):
        output = self.linear(x)
        output = output.view(-1, self.out_feature, self.pool_size)
        output = output.max(2)[0]

        return output


class Multihead_Attention(nn.Module):
    """
    Multi-head Attention
    """

    def __init__(self, model_dim, head_count=8, dropout=0.1):
        """
        initialization for variables and functions
        :param model_dim: hidden size
        :param head_count: head number, default 8
        :param dropout: dropout probability
        """
        super(Multihead_Attention, self).__init__()

        self.head_dim = model_dim // head_count
        self.model_dim = model_dim
        self.head_count = head_count
        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.head_dim)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.head_dim)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.head_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)
        self.sigmoid = nn.Hardtanh(min_val=0)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None, tau=None, Bernoulli=False):
        """
        run multi-head attention
        :param key: key, [batch, len, size]
        :param value: value, [batch, len, size]
        :param query: query, [batch, len, size]
        :param mask: mask
        :param layer_cache: layer cache for transformer decoder
        :param type: "self" or "context"
        :param tau: temperature, will be deprecated
        :param Bernoulli: use Bernoulli selection or not
        :return: attention output and attention weights
        """
        batch_size = key.size(0)
        head_dim = self.head_dim
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, head_dim) \
                .transpose(1, 2)    # [batch, head, len, head_dim]

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * head_dim)    # [batch, len, size]

        # For transformer decoder.
        # denote the device for multi-gpus
        device = str(query.device)
        if layer_cache is not None:
            # decoder self attention in the inference stage
            if type == "self":
                query, key, value = self.linear_query(query),\
                    self.linear_keys(query),\
                    self.linear_values(query)   # [batch, len, size]
                key = shape(key)    # [batch, head, k_len, head_dim]
                value = shape(value)    # [batch, head, v_len, head_dim]
                # print device and layer cache for debugging
                # print(device, layer_cache.keys())
                # decoding after the first step, with layer cache
                if layer_cache is not None:
                    # device name for correct saving
                    if layer_cache["self_keys_" + device] is not None:
                        key = torch.cat(
                            (layer_cache["self_keys_" + device], key),
                            dim=2)  # [batch, head, k_len, head_dim]
                    if layer_cache["self_values_" + device] is not None:
                        value = torch.cat(
                            (layer_cache["self_values_" + device], value),
                            dim=2)  # [batch, head, v_len, head_dim]
                    layer_cache["self_keys_" + device] = key
                    layer_cache["self_values_" + device] = value
            # decoder context attention in the inference stage
            elif type == "context":
                query = self.linear_query(query)
                # decoding after the first step, with layer cache
                if layer_cache is not None:
                    # device name for correct saving
                    if layer_cache["memory_keys_" + device] is None:
                        key, value = self.linear_keys(key),\
                            self.linear_values(value)   # [batch, len, size]
                        key = shape(key)    # [batch, head, k_len, head_dim]
                        value = shape(value)    # [batch, head, v_len, head_dim]
                    else:
                        key, value = layer_cache["memory_keys_" + device],\
                            layer_cache["memory_values_" + device]
                    layer_cache["memory_keys_" + device] = key
                    layer_cache["memory_values_" + device] = value
                else:
                    key, value = self.linear_keys(key),\
                        self.linear_values(value)   # [batch, len, size]
                    key = shape(key)    # [batch, head, k_len, head_dim]
                    value = shape(value)    # [batch, head, v_len, head_dim]
        else:
            query, key, value = self.linear_query(query),\
                self.linear_keys(key),\
                self.linear_values(value)   # [batch, len, size]
            key = shape(key)    # [batch, head, k_len, head_dim]
            value = shape(value)    # [batch, head, v_len, head_dim]

        query = shape(query)    # [batch, head, q_len, head_dim]

        key_len = key.size(2)
        query_len = query.size(2)

        if tau is not None:
            tau = tau.view(batch_size, -1, head_count,
                           1).transpose(1, 2)   # [batch, head, len, 1]
            tau_len = tau.size(2)
            assert query_len == tau_len
            query = query / (head_dim**tau)
        else:
            query = query / math.sqrt(head_dim)

        scores = torch.matmul(query, key.transpose(2, 3))   # [batch, head, q_len, k_len]

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # use Bernoulli selection or not
        if type == "context" and Bernoulli:
            attn = self.sigmoid(scores)  # [batch, head, q_len, k_len]
        else:
            attn = self.softmax(scores)  # [batch, head, q_len, k_len]

        drop_attn = self.dropout(attn)  # [batch, head, q_len, k_len]
        context = unshape(torch.matmul(drop_attn, value))   # [batch, q_len, size]

        output = self.final_linear(context)  # [batch, q_len, size]

        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous()   # [batch, q_len, k_len]

        return output, top_attn
