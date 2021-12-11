import math
import multiprocessing
import os
import queue
import subprocess
from argparse import ArgumentParser, Namespace

import torch
import utils
import yaml
from dataset import load_data
from tqdm import tqdm
from train import build_model
from utils import misc_utils


class DescriptionGenerator(object):
    def __init__(self, config, **opt):
        # Load config used for training and merge with testing options
        self.config = yaml.load(open(config, "r"))
        self.config = Namespace(**{**self.config, **opt})

        # Load training data.pkl for src and tgt vocabs
        self.data = load_data(self.config)

        # Load trained model checkpoints
        device, devices_ids = misc_utils.set_cuda(self.config)
        self.model, _ = build_model(None, self.config, device)
        self.model.eval()

    def predict(self, original_src: list) -> list:
        src_vocab = self.data["src_vocab"]
        tgt_vocab = self.data["tgt_vocab"]
        srcIds = src_vocab.convertToIdx(list(original_src), utils.UNK_WORD)
        src = torch.LongTensor(srcIds).unsqueeze(0)
        src_len = torch.LongTensor([len(srcIds)])

        if self.config.use_cuda:
            src = src.cuda()
            src_len = src_len.cuda()

        with torch.no_grad():

            if self.config.beam_size > 1:
                samples, alignments = self.model.beam_sample(
                    src, src_len, beam_size=self.config.beam_size, eval_=False
                )
            else:
                samples, alignments = self.model.sample(src, src_len)

        assert len(samples) == 1
        candidates = [tgt_vocab.convertToLabels(samples[0], utils.EOS)]

        # Replace unk with src tokens
        if self.config.unk and self.config.attention != "None":
            s = original_src
            c = candidates[0]
            align = alignments[0]
            cand = []
            for word, idx in zip(c, align):
                if word == utils.UNK_WORD and idx < len(s):
                    try:
                        cand.append(s[idx])
                    except:
                        cand.append(word)
                        print("%d %d\n" % (len(s), idx))
                else:
                    cand.append(word)
        return cand


class DescriptionGeneratorProxy(object):
    @staticmethod
    def enqueue_output(out, queue):
        for line in iter(out.readline, b""):
            queue.put(line)
        out.close()

    def __init__(self, gpu_id):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.process = subprocess.Popen(
            ["python", "api.py"],
            env=env,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            universal_newlines=True,
        )
        self.stdout_reader = multiprocessing.Queue()
        self.stdout_reader_process = multiprocessing.Process(
            target=DescriptionGeneratorProxy.enqueue_output,
            args=(self.process.stdout, self.stdout_reader),
            daemon=True,
        )
        self.stdout_reader_process.start()

    def send(self, src):
        self.process.stdin.write(f"{src}\n")
        self.process.stdin.flush()

    def recv(self, timeout=None):
        try:
            stdout = self.stdout_reader.get(timeout=timeout)
            return stdout.strip()
        except queue.Empty:
            return ""

    def flush(self):
        while True:
            line = self.recv()
            if line == "COMPLETE":
                break


class DescriptionGeneratorMultiprocessing(object):
    def __init__(self, n_gpus=8, n_process_per_gpu=8, **kwargs):
        self.proxies = []
        for gpu_id in range(n_gpus):
            for _ in range(n_process_per_gpu):
                self.proxies.append(DescriptionGeneratorProxy(gpu_id))
        for proxy in self.proxies:
            proxy.flush()

    def _predict_batch(self, src_list):
        """Batch size = n_gpu * n_process_per_gpu"""
        assert len(src_list) <= len(self.proxies)
        for proxy, src in zip(self.proxies, src_list):
            proxy.send(src)
        return [proxy.recv() for proxy, src in zip(self.proxies, src_list)]

    def predict_all(self, src_list):
        tgt_list = []
        for idx in tqdm(range(int(math.ceil(len(src_list) / len(self.proxies))))):
            tgt_list += self._predict_batch(
                src_list[idx * len(self.proxies) : (idx + 1) * len(self.proxies)]
            )
        return tgt_list


if __name__ == "__main__":
    g = DescriptionGenerator(
        config="yaml/title_summary_item_filter_t2t.yaml",
        gpus="0",
        restore=False,
        pretrain="experiments/3.8-finetune-big/best_bleu_checkpoint.pt",
        mode="eval",
        batch_size=1,
        beam_size=10,
        # refactor issue; workaround; delete afterwards:
        scale=1,
        char=False,
        use_cuda=True,
        seed=1234,
        model="tensor2tensor",
    )
    # For testing
    print("".join(g.predict(list("这东西真智障"))))
    # Interactive interface for multiprocessing
    print("COMPLETE")
    while True:
        src = input()
        print("".join(g.predict(list(src))))
