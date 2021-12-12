from typing import Tuple

import torch
import torch.nn as nn
from cached_property import cached_property
from torch import optim
from torch.nn.modules.transformer import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from webdataset.iterators import batched

from kobe.data.dataset import Batched, EncodedBatch
from kobe.data.vocab import BOS_ID, EOS_ID, PAD_ID
from kobe.utils import helpers


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(d_model)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        seq_length = x.shape[0]
        position_ids = (
            torch.arange(seq_length, device=x.device).unsqueeze(1).expand(x.shape[:2])
        )
        pe = self.pe(position_ids)
        x = x + pe
        x = self.layernorm(x)
        return self.dropout(x)


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
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout, norm_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        self.mode = mode

    @cached_property
    def device(self):
        return list(self.parameters())[0].device

    def forward(self, batched: Batched) -> EncodedBatch:
        src, src_key_padding_mask = Encoder._get_input(batched, self.mode)
        src = self.input_embedding(src)
        src = self.pos_encoder(src)
        token_encodings = self.transformer.forward(
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

    def configure_optimizers(self, lr: float):
        optimizer = optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-9)
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=16000
        )
        return [optimizer], [scheduler]


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
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, 4 * d_model, dropout, norm_first=True
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, batch: Batched, encoded_batch: EncodedBatch) -> torch.Tensor:
        tgt = self.embedding(batch.description_token_ids)
        tgt_mask = Decoder.generate_square_subsequent_mask(tgt.shape[0], tgt.device)
        outputs = self.decoder(
            tgt=tgt,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=batch.description_token_ids_mask,
            memory=encoded_batch.context_encodings,
            memory_key_padding_mask=encoded_batch.context_encodings,
        )
        return self.output(outputs)

    def predict(
        self,
        batch: Batched,
        encoded_batch: EncodedBatch,
        logits: torch.Tensor,
        beam_size: int = 0,
    ):
        if beam_size == 0:
            return self.greedy_decode(batch, encoded_batch, logits)
        else:
            raise NotImplementedError

    def greedy_decode(
        self, batch: Batched, encoded_batch: EncodedBatch, logits: torch.Tensor
    ):
        batch_size = encoded_batch.context_encodings.shape[1]
        tgt = self.embedding(torch.tensor([BOS_ID] * batch_size).unsqueeze(dim=0))
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

            pred_step = logits.argmax().tolist()
            for b in range(batch_size):
                if pred_all and pred_all[-1][b].item() == EOS_ID:
                    pred_step[b] = PAD_ID
            if all([pred == PAD_ID for pred in pred_step]):
                break
            pred_step = torch.tensor(pred_step)
            pred_all.append(pred_step)

            if idx < self.max_seq_len - 1:
                tgt_step = self.embedding(pred_step.unsqueeze(dim=0))
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
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
