import os
import random
import sys
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def set_cuda(config):
    use_cuda = torch.cuda.is_available() and len(config.gpus) > 0
    assert config.use_cuda == use_cuda
    if use_cuda:
        # torch.cuda.set_device(config.gpus[0])
        if len(config.gpus) > 1:
            torch.cuda.manual_seed_all(config.seed)
        else:
            torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
    device = (
        torch.device("cuda:{}".format(config.gpus[0]))
        if use_cuda
        else torch.device("cpu")
    )
    devices_ids = config.gpus
    return device, devices_ids


def set_tensorboard(config):
    summary_dir = os.path.join(config.logdir, config.expname)
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    for file_name in os.listdir(summary_dir):
        if file_name.startswith("events.out.tfevents"):
            print(f"Event file {file_name} already exists")
            if input("Remove this file? (y/n) ") == "y":
                os.remove(os.path.join(summary_dir, file_name))
                print(f"Event file {file_name} removed")
    return SummaryWriter(summary_dir)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
