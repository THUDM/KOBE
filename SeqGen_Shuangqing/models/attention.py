import math
import torch
import torch.nn as nn


class luong_attention(nn.Module):

    def __init__(self, hidden_size, emb_size, pool_size=0):
        super(luong_attention, self).__init__()
        self.hidden_size, self.emb_size, self.pool_size = hidden_size, emb_size, pool_size
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        if pool_size > 0:
            self.linear_out = maxout(
                2*hidden_size + emb_size, hidden_size, pool_size)
        else:
            self.linear_out = nn.Sequential(
                nn.Linear(2*hidden_size + emb_size, hidden_size), nn.Tanh())
        self.softmax = nn.Softmax(dim=1)

    def init_context(self, context):
        self.context = context.transpose(0, 1)

    def forward(self, h, x):
        gamma_h = self.linear_in(h).unsqueeze(2)    # batch * size * 1
        weights = torch.bmm(self.context, gamma_h).squeeze(2)   # batch * time
        weights = self.softmax(weights)   # batch * time
        c_t = torch.bmm(weights.unsqueeze(
            1), self.context).squeeze(1)  # batch * size
        output = self.linear_out(torch.cat([c_t, h, x], 1))

        return output, weights


class luong_gate_attention(nn.Module):

    def __init__(self, hidden_size, emb_size, prob=0.1):
        super(luong_gate_attention, self).__init__()
        self.hidden_size, self.emb_size = hidden_size, emb_size
        self.linear_enc = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob),
                                        nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.linear_in = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob),
                                       nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.linear_out = nn.Sequential(nn.Linear(2*hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob),
                                        nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=prob)

    def init_context(self, context):
        self.context = context.transpose(0, 1)

    def forward(self, h, selfatt=False):
        if selfatt:
            # Batch_size * Length * Hidden_size
            gamma_enc = self.linear_enc(self.context)
            # Batch_size * Hidden_size * Length
            gamma_h = gamma_enc.transpose(1, 2)
            # Batch_size * Length * Length
            weights = torch.bmm(gamma_enc, gamma_h)
            weights = self.softmax(weights/math.sqrt(512))
            # Batch_size * Length * Hidden_size
            c_t = torch.bmm(weights, gamma_enc)
            output = self.linear_out(
                torch.cat([gamma_enc, c_t], 2)) + self.context
            # Length * Batch_size * Hidden_size
            output = output.transpose(0, 1)
        else:
            gamma_h = self.linear_in(h).unsqueeze(2)
            weights = self.dropout(torch.bmm(self.context, gamma_h).squeeze(2))
            weights = self.softmax(weights)
            c_t = torch.bmm(weights.unsqueeze(1), self.context).squeeze(1)
            output = self.linear_out(torch.cat([h, c_t], 1))

        return output, weights


class bahdanau_attention(nn.Module):

    def __init__(self, hidden_size, emb_size, pool_size=0):
        super(bahdanau_attention, self).__init__()
        self.linear_encoder = nn.Linear(hidden_size, hidden_size)
        self.linear_decoder = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, 1)
        self.linear_r = nn.Linear(hidden_size*2+emb_size, hidden_size*2)
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def init_context(self, context):
        self.context = context.transpose(0, 1)

    def forward(self, h, x):
        gamma_encoder = self.linear_encoder(
            self.context)           # batch * time * size
        gamma_decoder = self.linear_decoder(
            h).unsqueeze(1)    # batch * 1 * size
        weights = self.linear_v(
            self.tanh(gamma_encoder+gamma_decoder)).squeeze(2)   # batch * time
        weights = self.softmax(weights)   # batch * time
        c_t = torch.bmm(weights.unsqueeze(
            1), self.context).squeeze(1)  # batch * size
        r_t = self.linear_r(torch.cat([c_t, h, x], dim=1))
        output = r_t.view(-1, self.hidden_size, 2).max(2)[0]

        return output, weights


class maxout(nn.Module):

    def __init__(self, in_feature, out_feature, pool_size):
        super(maxout, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.pool_size = pool_size
        self.linear = nn.Linear(in_feature, out_feature*pool_size)

    def forward(self, x):
        output = self.linear(x)
        output = output.view(-1, self.out_feature, self.pool_size)
        output = output.max(2)[0]

        return output


class Multihead_Attention(nn.Module):

    def __init__(self, model_dim, head_count=8, dropout=0.1):
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

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None, tau=None):
        batch_size = key.size(0)
        head_dim = self.head_dim
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, head_dim) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * head_dim)

        # For transformer decoder.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query),\
                    self.linear_keys(query),\
                    self.linear_values(query)
                key = shape(key)
                value = shape(value)
                if layer_cache is not None:
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat(
                            (layer_cache["self_keys"], key),
                            dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat(
                            (layer_cache["self_values"], value),
                            dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key, value = self.linear_keys(key),\
                            self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache["memory_keys"],\
                            layer_cache["memory_values"]
                    layer_cache["memory_keys"] = key
                    layer_cache["memory_values"] = value
                else:
                    key, value = self.linear_keys(key),\
                        self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
        else:
            query, key, value = self.linear_query(query),\
                self.linear_keys(key),\
                self.linear_values(value)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        if tau is not None:
            tau = tau.view(batch_size, -1, head_count,
                           1).transpose(1, 2)
            tau_len = tau.size(2)
            assert query_len == tau_len
            query = query / (head_dim**tau)
        else:
            query = query / math.sqrt(head_dim)

        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            if tau is not None:
                scores = scores.masked_fill(mask, -1e10)
            else:
                scores = scores.masked_fill(mask, -1e18)

        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = unshape(torch.matmul(drop_attn, value))

        output = self.final_linear(context)

        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous()

        return output, top_attn
