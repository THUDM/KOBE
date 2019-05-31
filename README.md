# KOBE

### [Project](https://sites.google.com/view/kobe2019) | [arXiv](https://arxiv.org/abs/1903.12457)

Towards **K**n**O**wledge-**B**ased p**E**rsonalized Product Description Generation in E-commerce.<br>
[Qibin Chen](https://www.qibin.ink)<sup>\*</sup>, [Junyang Lin](https://justinlin610.github.io)<sup>\*</sup>, Yichang Zhang, [Hongxia Yang](https://sites.google.com/site/hystatistics/home), [Jingren Zhou](http://www.cs.columbia.edu/~jrzhou/), [Jie Tang](http://keg.cs.tsinghua.edu.cn/jietang/).<br>
<sup>*</sup>Equal contribution.<br>
In KDD 2019 (Applied Data Science Track)

## Prerequisites

- Linux or macOS
- Python 3
- PyTorch >= 1.0.1
- NVIDIA GPU + CUDA cuDNN

## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/THUDM/KOBE
cd KOBE
```

Please install dependencies by

```bash
pip install -r requirements.txt
```

### Dataset

- TaoDescribe
    - We use the **TaoDescribe** dataset, which contains 2,129,187 product titles and descriptions in Chinese.
    - (optional) You can download the un-preprocessed dataset from [here](https://www.dropbox.com/sh/nnnq9eobmn6u44v/AAA7s4YkVbslS-6slDIOn4MYa) or [here](https://tianchi.aliyun.com/dataset/dataDetail?dataId=9717).

### Training
#### Download preprocessed data

- First, download the preprocessed TaoDescribe dataset by running `python scripts/download_preprocessed_tao.py`.
    - If you're in regions where Dropbox are blocked (e.g. Mainland China), try `python scripts/download_preprocessed_tao.py --cn`.
- (optional) You can peek into the `data/aspect-user/preprocessed/test.src.str` and `data/aspect-user/preprocessed/test.tgt.str`, which includes the product titles and descriptions in the test set, respectively. `<4> <a>` means this product is intended to show with aspect `<a>` and user category `<4>`. Note: this slightly differs from the `<A-1>`, `<U-1>` format descripted in the paper but basically they are the same thing.

#### Start training

- Different configurations for models in the paper are stored under the `configs/` directory. Launch a specific experiment with `--config` to specify the path to your desired model setting and `--expname` to specify the name/number of this experiment which will be used in logging.
- We include three config files: the baseline, KOBE without adding external knowledge, and full KOBE model.

- Baseline

```bash
python core/train.py --config configs/baseline.yaml --expname baseline
```

- KOBE without adding knowledge

```bash
python core/train.py --config configs/aspect_user.yaml --expname aspect-user
```

- KOBE

```bash
python core/train.py --config configs/aspect_user_knowledge.yaml --expname aspect-user-knowledge
```

The default `batch size` is set to 64.
If you are having OOM problems, try to decrease it with the flag `--batch-size`.

#### Track training progress

- Tensorboard

```bash
tensorboard --logdir experiments --port 6006
```

### Generation

TODO

### Evaluation

TODO

If you have ANY difficulties to get things working, feel free to open an issue.
You can expect a reply within 24 hours.

## Cite

Please cite our paper if you use this code in your own work:

```
@article{chen2019towards,
  title={Towards Knowledge-Based Personalized Product Description Generation in E-commerce},
  author={Chen, Qibin and Lin, Junyang and Zhang, Yichang and Yang, Hongxia and Zhou, Jingren and Tang, Jie},
  journal={arXiv preprint arXiv:1903.12457},
  year={2019}
}
```

