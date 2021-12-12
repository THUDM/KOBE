import argparse
import random

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from kobe.data.dataset import KobeDataModule
from kobe.models.model import KobeModel
from kobe.utils.options import add_args, add_options

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    add_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    wandb_logger = WandbLogger(name=args.name, project="kobe", log_model=True)
    # overwrite specified args for hyperparam sweeping
    for k, v in wandb_logger.experiment.config.items():
        setattr(args, k, v)
    wandb_logger.log_hyperparams(args)

    dm = KobeDataModule(
        args.train_data,
        args.valid_data,
        args.test_data,
        args.text_vocab_path,
        args.max_seq_len,
        args.batch_size,
        args.num_workers,
    )

    model = KobeModel(args)
    if args.load_file is not None:
        assert args.test
        model = model.load_from_checkpoint(args.load_file, args=args, dm=dm)

    callbacks = [
        EarlyStopping(monitor=f"val/bleu", mode="max", patience=args.patience),
        ModelCheckpoint(monitor=f"val/bleu", mode="max"),
        TQDMProgressBar(refresh_rate=10),
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=wandb_logger,
        gpus=args.gpu,
        auto_select_gpus=True,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
    )

    if not args.test:
        trainer.fit(model, datamodule=dm)
        # will automatically load and test the best checkpoint instead of the last model
        trainer.test(datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
