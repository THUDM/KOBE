from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import sentencepiece as spm
import torch
import torch.nn as nn
from bert_score import BERTScorer
from sacrebleu.metrics.bleu import BLEU, _get_tokenizer
from torch import optim
from torch.nn.init import xavier_uniform_
from transformers.models.bert.tokenization_bert import BertTokenizer

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
        self._reset_parameters()

        self.decoding_strategy = args.decoding_strategy
        self.vocab = BertTokenizer.from_pretrained(args.text_vocab_path)
        self.bleu = BLEU(tokenize=args.tokenize)
        self.sacre_tokenizer = _get_tokenizer(args.tokenize)()
        self.bert_scorer = BERTScorer(lang=args.tokenize, rescale_with_baseline=True)

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

        preds = self.decoder.predict(
            encoded_batch=encoded, decoding_strategy=self.decoding_strategy
        )
        generated = self.vocab.batch_decode(preds.T.tolist(), skip_special_tokens=True)

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

        generated = [g for o in outputs for g in o.generated]
        references = [r for o in outputs for r in o.descriptions]

        # fmt: off
        # BLEU score
        self.log(f"{prefix}/bleu", self.bleu.corpus_score(generated, [references]).score)

        # Diversity score
        self.log(f"{prefix}/diversity_3", float(helpers.diversity([self.sacre_tokenizer(g) for g in generated], n=3)))
        self.log(f"{prefix}/diversity_4", float(helpers.diversity([self.sacre_tokenizer(g) for g in generated], n=4)))
        self.log(f"{prefix}/diversity_5", float(helpers.diversity([self.sacre_tokenizer(g) for g in generated], n=5)))
        # fmt: on

        # BERTScore
        p, r, f = self.bert_scorer.score(generated, references)
        self.log(f"{prefix}/BERTScore_P", p.mean().item())
        self.log(f"{prefix}/BERTScore_R", r.mean().item())
        self.log(f"{prefix}/BERTScore_F", f.mean().item())

        # Examples
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
