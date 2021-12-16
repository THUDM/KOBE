from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import sentencepiece as spm
import torch
import torch.nn as nn
from sacrebleu.metrics import BLEU
from torch import optim
from torch.nn.init import xavier_uniform_

import wandb
from kobe.data.dataset import Batched, DecodedBatch
from kobe.models.scheduler import WarmupDecayLR
from kobe.models.transformer import Decoder, Encoder
from kobe.utils import helpers


class KobeModel(pl.LightningModule):
    def __init__(self, args):
        super(KobeModel, self).__init__()

        self.encoder = Encoder(
            vocab_size=args.text_vocab_size + args.cond_vocab_size,
            max_seq_len=args.max_seq_len,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_encoder_layers,
            dropout=args.dropout,
            mode=args.mode,
        )
        self.decoder = Decoder(
            vocab_size=args.text_vocab_size,
            max_seq_len=args.max_seq_len,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_decoder_layers,
            dropout=args.dropout,
        )
        self.lr = args.lr
        self.d_model = args.d_model
        self.loss = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=0, label_smoothing=0.1
        )
        self.bleu = BLEU()
        self.vocab = spm.SentencePieceProcessor()
        self.vocab.Load(args.text_vocab_path)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def _tokenwise_loss_acc(
        self, logits: torch.Tensor, batch: Batched
    ) -> Tuple[torch.Tensor, float]:
        unmask = ~batch.description_token_ids_mask.T[1:]
        unmasked_logits = logits[unmask]
        unmasked_targets = batch.description_token_ids[1:][unmask]
        acc = helpers.accuracy(unmasked_logits, unmasked_targets)
        return self.loss(logits.transpose(1, 2), batch.description_token_ids[1:]), acc

    def training_step(self, batch: Batched, batch_idx: int):
        encoded = self.encoder.forward(batch)
        logits = self.decoder.forward(batch, encoded)
        loss, acc = self._tokenwise_loss_acc(logits, batch)
        self.lr_schedulers().step()
        self.log("train/loss", loss.item())
        self.log("train/acc", acc)
        return loss

    def _shared_eval_step(self, batch: Batched, batch_idx: int) -> DecodedBatch:
        encoded = self.encoder.forward(batch)
        logits = self.decoder.forward(batch, encoded)
        loss, acc = self._tokenwise_loss_acc(logits, batch)

        preds = self.decoder.predict(encoded_batch=encoded, beam_size=0)
        generated = self.vocab.Decode(preds.T.tolist())

        return DecodedBatch(
            loss=loss.item(),
            acc=acc,
            generated=generated,
            descriptions=batch.descriptions,
        )

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_eval_step(batch, batch_idx)

    def _shared_epoch_end(self, outputs: List[DecodedBatch], prefix):
        loss = np.mean([o.loss for o in outputs])
        acc = np.mean([o.acc for o in outputs])
        self.log(f"{prefix}/loss", loss)
        self.log(f"{prefix}/acc", acc)

        generated = [" ".join(g) for o in outputs for g in o.generated]
        references = [" ".join(g) for o in outputs for g in o.descriptions]

        def bleu_with_trunc_length(max_char_len: int):
            trunc_generated = [
                " ".join(g[:max_char_len]) for o in outputs for g in o.generated
            ]
            return self.bleu.corpus_score(trunc_generated, [references]).score

        for max_char_len in range(32, 257, 32):
            self.log(
                f"{prefix}/bleu_{max_char_len}", bleu_with_trunc_length(max_char_len)
            )
        self.log(f"{prefix}/bleu", bleu_with_trunc_length(150))

        columns = ["Generated", "Reference"]
        data = list(zip(generated[:256:16], references[:256:16]))
        table = wandb.Table(data=data, columns=columns)
        self.logger.experiment.log({f"examples/{prefix}": table})

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.98))
        scheduler = WarmupDecayLR(optimizer, warmup_steps=10000, d_model=self.d_model)
        return [optimizer], [scheduler]
