import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupDecayLR(_LRScheduler):
    """Linear warmup + inverse square root decay.

    Described in the T5 paper https://arxiv.org/abs/1910.10683.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        d_model: int,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # scale = min(
        #     self.last_epoch * self.warmup_steps ** -1.5,
        #     max(self.last_epoch, self.warmup_steps) ** -0.5,
        # )
        epoch = self.last_epoch + 1
        scale = self.d_model ** (-0.5) * min(
            epoch ** (-0.5), epoch * self.warmup_steps ** (-1.5)
        )
        return [base_lr * scale for base_lr in self.base_lrs]
