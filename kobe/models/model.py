from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sacrebleu.metrics import BLEU
from torch import optim

from kobe.data.dataset import Batched, DecodedBatch
from kobe.models.transformer import Decoder, Encoder
from kobe.utils import helpers


class KobeModel(pl.LightningModule):
    def __init__(self, args):
        super(KobeModel, self).__init__()

        self.encoder = Encoder(
            vocab_size=args.vocab_size,
            max_seq_len=args.max_seq_len,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_encoder_layers,
            dropout=args.dropout,
            mode=args.model,
        )
        self.decoder = Decoder(
            vocab_size=args.vocab_size,
            max_seq_len=args.max_seq_len,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_decoder_layers,
            dropout=args.dropout,
        )
        self.lr = args.lr
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.bleu = BLEU()
        self.vocab = spm.SentencePieceProcessor()
        self.vocab.Load(args.text_vocab_path)

    def _tokenwise_loss_acc(
        self, logits: torch.Tensor, batch: Batched
    ) -> Tuple[torch.Tensor, float]:
        unmask = ~batch.description_token_ids_mask.T[1:]
        unmasked_logits = logits[:-1][unmask]
        unmasked_targets = batch.description_token_ids[1:][unmask]
        acc = helpers.accuracy(unmasked_logits, unmasked_targets)
        return self.loss(unmasked_logits, unmasked_targets).mean(), acc

    def training_step(self, batch: Batched, batch_idx: int):
        encoded = self.encoder.forward(batch)
        logits = self.decoder.forward(batch, encoded)
        loss, acc = self._tokenwise_loss_acc(logits, batch)
        self.log("train/loss", loss.item())
        self.log("train/acc", acc)
        return loss

    def _shared_eval_step(self, batch: Batched, batch_idx: int) -> DecodedBatch:
        encoded = self.encoder.forward(batch)
        logits = self.decoder.forward(batch, encoded)
        loss = self.loss(encoded, logits)

        preds = self.decoder.predict(encoded_batch=encoded, beam_size=0)
        generated = self.vocab.Decode(preds.T.tolist())

        return DecodedBatch(
            loss=loss.item(), generated=generated, descriptions=batch.descriptions
        )

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_eval_step(batch, batch_idx)

    def _shared_epoch_end(self, outputs: List[DecodedBatch], prefix):
        loss = np.mean([o.loss for o in outputs])
        self.log(f"{prefix}/loss", loss)

        generated = [" ".join(g) for o in outputs for g in o.generated]
        references = [" ".join(g) for o in outputs for g in o.descriptions]
        bleu = self.bleu.corpus_score(generated, [references])
        self.log(f"{prefix}/bleu", bleu.score)

        columns = ["Generated", "Reference"]
        data = [generated[:256:16], references[:256:16]]
        table = wandb.Table(data=data, columns=columns)

        self.log(f"{prefix}/examples", table)

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-9
        )
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=16000
        )
        return [optimizer], [scheduler]
