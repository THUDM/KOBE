import glob
from dataclasses import dataclass
from typing import List

import pytorch_lightning as pl
import sentencepiece as spm
import torch
import webdataset as wds
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader


@dataclass
class Example:
    title_token_ids: List[int]
    description_token_ids: List[int]
    condition_token_ids: List[int]
    fact_token_ids: List[int]
    description: str


@dataclass
class TensorDict:
    def detach(self):
        detached_dict = {
            field: getattr(self, field).detach()
            if isinstance(getattr(self, field), torch.Tensor)
            else getattr(self, field)
            for field in self.__dataclass_fields__
        }
        return self.__class__(**detached_dict)

    def cpu(self):
        detached_dict = {
            field: getattr(self, field).cpu()
            if isinstance(getattr(self, field), torch.Tensor)
            else getattr(self, field)
            for field in self.__dataclass_fields__
        }
        return self.__class__(**detached_dict)


@dataclass
class Batched(TensorDict):
    # Source
    title_token_ids: torch.Tensor
    title_token_ids_mask: torch.Tensor
    # Attribute Fusion
    cond_title_token_ids: torch.Tensor
    cond_title_token_ids_mask: torch.Tensor
    # Knowledge Incorporation
    fact_token_ids: torch.Tensor
    fact_token_ids_mask: torch.Tensor
    # Attribute Fusion + Knowledge Incorporation
    cond_title_fact_token_ids: torch.Tensor
    cond_title_fact_token_ids_mask: torch.Tensor
    # Target
    description_token_ids: torch.Tensor
    description_token_ids_mask: torch.Tensor
    descriptions: List[str]


def from_processed(url: str):
    urls = sorted(glob.glob(url))
    return wds.WebDataset(urls).decode().map(lambda d: Example(**d["json"]))


def get_collate_fn(text_vocab_size: int, max_seq_length: int):
    def collate_fn(examples: List[Example]) -> Batched:
        from kobe.data.vocab import BOS_ID, EOS_ID

        title_token_ids = pad_sequence(
            [
                torch.tensor(
                    [BOS_ID] + e.title_token_ids[: max_seq_length - 2] + [EOS_ID]
                )
                for e in examples
            ]
        )
        fact_token_ids = pad_sequence(
            [
                torch.tensor(
                    [BOS_ID] + e.fact_token_ids[: max_seq_length - 2] + [EOS_ID]
                )
                for e in examples
            ]
        )
        description_token_ids = pad_sequence(
            [
                torch.tensor(
                    [BOS_ID] + e.description_token_ids[: max_seq_length - 2] + [EOS_ID]
                )
                for e in examples
            ]
        )
        cond_title_token_ids = pad_sequence(
            [
                torch.tensor(
                    (
                        [BOS_ID]
                        + [
                            cond_id + text_vocab_size
                            for cond_id in e.condition_token_ids
                        ]
                        + e.title_token_ids
                    )[: max_seq_length - 1]
                    + [EOS_ID]
                )
                for e in examples
            ]
        )
        cond_title_fact_token_ids = pad_sequence(
            [
                torch.tensor(
                    (
                        [BOS_ID]
                        + [
                            cond_id + text_vocab_size
                            for cond_id in e.condition_token_ids
                        ]
                        + e.title_token_ids
                        + [EOS_ID]
                        + e.fact_token_ids
                    )[: max_seq_length - 1]
                    + [EOS_ID]
                )
                for e in examples
            ]
        )
        descriptions = [e.description for e in examples]

        return Batched(
            title_token_ids=title_token_ids,
            title_token_ids_mask=(title_token_ids == 0).T,
            fact_token_ids=fact_token_ids,
            fact_token_ids_mask=(fact_token_ids == 0).T,
            cond_title_token_ids=cond_title_token_ids,
            cond_title_token_ids_mask=(cond_title_token_ids == 0).T,
            description_token_ids=description_token_ids,
            description_token_ids_mask=(description_token_ids == 0).T,
            cond_title_fact_token_ids=cond_title_fact_token_ids,
            cond_title_fact_token_ids_mask=(cond_title_fact_token_ids == 0).T,
            descriptions=descriptions,
        )

    return collate_fn


class KobeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data: str,
        valid_data: str,
        test_data: str,
        vocab_path: str,
        max_seq_length: int,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        text_tokenizer = spm.SentencePieceProcessor()
        text_tokenizer.Load(vocab_path)
        self.text_vocab_size = len(text_tokenizer)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = from_processed(self.train_data)
            self.valid = from_processed(self.valid_data)
        if stage == "test" or stage is None:
            self.test = from_processed(self.test_data)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=get_collate_fn(self.text_vocab_size, self.max_seq_length),
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=get_collate_fn(self.text_vocab_size, self.max_seq_length),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=get_collate_fn(self.text_vocab_size, self.max_seq_length),
        )


if __name__ == "__main__":
    dm = KobeDataModule(
        train_data="data-v2/processed/train-*.tar",
        valid_data="data-v2/processed/valid-*.tar",
        test_data="data-v2/processed/test-*.tar",
        vocab_path="data-v2/vocab.text.model",
        max_seq_length=512,
        batch_size=32,
        num_workers=8,
    )
    dm.setup(stage="fit")
    for batch in dm.val_dataloader():
        print(batch)
        import pdb

        pdb.set_trace()
