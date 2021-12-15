import math
from typing import Tuple

import torch
import torch.nn as nn
from cached_property import cached_property
from torch.nn.modules.transformer import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

from kobe.data.dataset import Batched, EncodedBatch
from kobe.data.vocab import BOS_ID, EOS_ID, PAD_ID
from kobe.utils import helpers


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        """
        initialization of required variables and functions
        :param dropout: dropout probability
        :param dim: hidden size
        :param max_len: maximum length
        """
        super(PositionalEncoding, self).__init__()
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
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb):
        """
        create positional encoding
        :param emb: word embedding
        :param step: step for decoding in inference
        :return: positional encoding representation
        """
        emb *= math.sqrt(self.dim)
        emb = emb + self.pe[: emb.size(0)]  # [len, batch, size]
        emb = self.dropout(emb)
        return emb


class Encoder(nn.Module):
    @staticmethod
    def from_args(args) -> "Encoder":
        return Encoder(
            args.text_vocab_size + args.cond_vocab_size,
            args.max_seq_len,
            args.d_model,
            args.nhead,
            args.num_encoder_layers,
            args.dropout,
            args.mode,
        )

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        mode: str,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(dropout, d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout, norm_first=True
        )
        self.encoder = TransformerEncoder(
            encoder_layer, num_layers, nn.LayerNorm(d_model)
        )
        self.mode = mode

    @cached_property
    def device(self):
        return list(self.parameters())[0].device

    def forward(self, batched: Batched) -> EncodedBatch:
        src, src_key_padding_mask = Encoder._get_input(batched, self.mode)
        src = self.input_embedding(src)
        src = self.pos_encoder(src)
        token_encodings = self.encoder.forward(
            src=src, src_key_padding_mask=src_key_padding_mask
        )
        return EncodedBatch(
            context_encodings=token_encodings,
            context_encodings_mask=src_key_padding_mask,
        )

    @staticmethod
    def _get_input(batched: Batched, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
        return {
            helpers.BASELINE: (batched.title_token_ids, batched.title_token_ids_mask),
            helpers.KOBE_ATTRIBUTE: (
                batched.cond_title_token_ids,
                batched.cond_title_token_ids_mask,
            ),
            helpers.KOBE_KNOWLEDGE: (
                batched.title_fact_token_ids,
                batched.title_fact_token_ids_mask,
            ),
            helpers.KOBE_FULL: (
                batched.cond_title_fact_token_ids,
                batched.cond_title_fact_token_ids_mask,
            ),
        }[mode]


class Decoder(nn.Module):
    @staticmethod
    def from_args(args) -> "Decoder":
        return Decoder(
            args.text_vocab_size,
            args.max_seq_len,
            args.d_model,
            args.nhead,
            args.num_encoder_layers,
            args.dropout,
        )

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
    ):
        super(Decoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(dropout, d_model)
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, 4 * d_model, dropout, norm_first=True
        )
        self.decoder = TransformerDecoder(
            decoder_layer, num_layers, nn.LayerNorm(d_model)
        )
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, batch: Batched, encoded_batch: EncodedBatch) -> torch.Tensor:
        tgt = self.embedding(batch.description_token_ids)
        tgt = self.pos_encoder(tgt)
        tgt_mask = Decoder.generate_square_subsequent_mask(tgt.shape[0], tgt.device)
        outputs = self.decoder(
            tgt=tgt,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=batch.description_token_ids_mask,
            memory=encoded_batch.context_encodings,
            memory_key_padding_mask=encoded_batch.context_encodings_mask,
        )
        return self.output(outputs)

    def predict(self, encoded_batch: EncodedBatch, beam_size: int):
        if beam_size == 0:
            return self.greedy_decode(encoded_batch)
        else:
            raise NotImplementedError

    def greedy_decode(self, encoded_batch: EncodedBatch):
        batch_size = encoded_batch.context_encodings.shape[1]
        tgt = self.embedding(
            torch.tensor(
                [BOS_ID] * batch_size, device=encoded_batch.context_encodings.device
            ).unsqueeze(dim=0)
        )
        tgt = self.pos_encoder(tgt)
        tgt_mask = Decoder.generate_square_subsequent_mask(self.max_seq_len, tgt.device)
        pred_all = []
        for idx in range(self.max_seq_len):
            outputs = self.decoder(
                tgt,
                tgt_mask=tgt_mask[: idx + 1, : idx + 1],
                memory=encoded_batch.context_encodings,
                memory_key_padding_mask=encoded_batch.context_encodings_mask,
            )
            logits = self.output(outputs[-1])

            pred_step = logits.argmax(dim=1).tolist()
            for b in range(batch_size):
                if pred_all and pred_all[-1][b].item() in [EOS_ID, PAD_ID]:
                    pred_step[b] = PAD_ID
            if all([pred == PAD_ID for pred in pred_step]):
                break
            pred_step = torch.tensor(pred_step, device=tgt.device)
            pred_all.append(pred_step)

            if idx < self.max_seq_len - 1:
                tgt_step = self.pos_encoder(self.embedding(pred_step.unsqueeze(dim=0)))
                tgt = torch.cat([tgt, tgt_step], dim=0)

        preds = torch.stack(pred_all)
        return preds

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        r"""
        Generate a square mask for the sequence. The masked positions are filled with
          float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        return torch.triu(
            torch.full((sz, sz), float("-inf"), device=device), diagonal=1
        )
