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
    - We use the [TaoDescribe](https://tianchi.aliyun.com/dataset/dataDetail?dataId=9717) dataset, which contains 2,129,187 product titles and descriptions in Chinese.
    - (optional) You can download the un-preprocessed dataset from [TODO]().

### Training
#### Download preprocessed dataset

- First, download the preprocessed TaoDescribe dataset by running `python scripts/download_preprocessed_tao.py`.
    - If you're in regions where Dropbox are blocked (e.g. Mainland China), try `python scripts/cn_download_preprocessed_tao.py`.

#### Start training

- TODO

#### Track training progress

- Tensorboard

### Generation

### Evaluation

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

