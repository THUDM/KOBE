import os
import pickle

import torch

import utils


def load_data(config):
    """
    load data.
    update "data" due to the saved path in the pickle file
    :return: a dict with data and vocabulary
    """
    print("loading data...\n")
    data = pickle.load(open(config.data + "data.pkl", "rb"))
    # retrieve data, due to the problem of path.
    data["train"]["length"] = int(data["train"]["length"] * config.scale)
    data["train"]["srcF"] = os.path.join(config.data, "train.src.id")
    data["train"]["original_srcF"] = os.path.join(config.data, "train.src.str")
    data["train"]["tgtF"] = os.path.join(config.data, "train.tgt.id")
    data["train"]["original_tgtF"] = os.path.join(config.data, "train.tgt.str")
    data["test"]["srcF"] = os.path.join(config.data, "test.src.id")
    data["test"]["original_srcF"] = os.path.join(config.data, "test.src.str")
    data["test"]["tgtF"] = os.path.join(config.data, "test.tgt.id")
    data["test"]["original_tgtF"] = os.path.join(config.data, "test.tgt.str")

    if config.knowledge:
        train_set = utils.BiKnowledgeDataset(
            os.path.join(config.data, 'train.supporting_facts'),
            infos=data['train'], char=config.char)
        valid_set = utils.BiKnowledgeDataset(
            os.path.join(config.data, 'test.supporting_facts'),
            infos=data['test'], char=config.char)
    else:
        train_set = utils.BiDataset(data["train"], char=config.char)
        valid_set = utils.BiDataset(data["test"], char=config.char)

    src_vocab = data["dict"]["src"]
    tgt_vocab = data["dict"]["tgt"]
    config.src_vocab_size = src_vocab.size()
    config.tgt_vocab_size = tgt_vocab.size()

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=utils.knowledge_padding if config.knowledge else utils.padding,
    )
    if hasattr(config, "valid_batch_size"):
        valid_batch_size = config.valid_batch_size
    else:
        valid_batch_size = config.batch_size
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_set,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.knowledge_padding if config.knowledge else utils.padding,
    )
    return {
        "train_set": train_set,
        "valid_set": valid_set,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
    }
