## [KOBE v2: Towards Knowledge-Based Personalized Product Description Generation in E-commerce](https://arxiv.org/abs/1903.12457)

[![Unittest](https://img.shields.io/github/workflow/status/THUDM/KOBE/Install)](https://github.com/THUDM/KOBE/actions/workflows/install.yml)
[![GitHub stars](https://img.shields.io/github/stars/THUDM/KOBE)](https://github.com/THUDM/KOBE/stargazers)
[![GitHub license](https://img.shields.io/github/license/THUDM/KOBE)](https://github.com/THUDM/KOBE/blob/master/LICENSE)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

**New:** We release **KOBE v2**, a refactored version of the original code with the latest deep learning tools in 2021 and greatly improved the installation, reproducibility, performance, and visualization, in memory of Kobe Bryant.

This repo contains code and pre-trained models for KOBE, a sequence-to-sequence based approach for automatically generating product descriptions by leveraging conditional inputs, e.g., user category, and incorporating knowledge with retrieval augmented product titles.

Paper accepted at KDD 2019 (Applied Data Science Track). Latest version at [arXiv](https://arxiv.org/abs/1903.12457).

- [KOBE v2: Towards Knowledge-Based Personalized Product Description Generation in E-commerce](#kobe-v2-towards-knowledge-based-personalized-product-description-generation-in-e-commerce)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Dataset](#dataset)
  - [Training](#training)
    - [Build vocabulary](#build-vocabulary)
    - [Preprocessing](#preprocessing)
    - [Start training](#start-training)
    - [Track training progress](#track-training-progress)
  - [Generation](#generation)
  - [Evaluation](#evaluation)
- [Cite](#cite)

## Prerequisites

- Linux
- Python >= 3.6
- PyTorch >= 1.10.
- PyTorch Lightning >= 1.5.4

## Getting Started

### Installation

Clone and install KOBE.

```bash
git clone https://github.com/THUDM/KOBE
cd KOBE
pip install -e .
```

Verify that KOBE is correctly installed by `import kobe`.

### Dataset

- We use the **TaoDescribe** dataset, which contains 2,129,187 product titles and descriptions in Chinese.
<!-- - (optional) You can download the un-preprocessed dataset from [here](https://www.dropbox.com/sh/nnnq9eobmn6u44v/AAA7s4YkVbslS-6slDIOn4MYa) or [here (for users in China)](https://tianchi.aliyun.com/dataset/dataDetail?dataId=9717). -->

<details>
<summary>
Meanings of downloaded data files
</summary>
<ul>
<li> train/valid/test.title: The product title as input (source) </li>
<li> train/valid/test.desc: The product description as output (generation target) </li>
<li> train/valid/test.cond: The product attribute and user category used as conditions in the KOBE model. The interpretations of these tags are explained at https://github.com/THUDM/KOBE/issues/14#issuecomment-516262659. </li>
<li> train/valid/test.fact: The retrieved knowledge for each product </li>
</ul>
</details>

<!-- - First, download the preprocessed TaoDescribe dataset by running `python scripts/download_preprocessed_tao.py`.
    - If you're in regions where Dropbox are blocked (e.g. Mainland China), try `python scripts/download_preprocessed_tao.py --cn`. -->

### Training
#### Build vocabulary

The first step is to build a vocabulary on the Chinese product title, product description and retrieved knowledge:

```bash
python -m kobe.vocab --input data-v2/raw/train.title data-v2/raw/train.desc data-v2/raw/train.fact --vocab-file data-v2/vocab.text
python -m kobe.vocab --input data-v2/raw/train.cond --vocab-file data-v2/vocab.cond --vocab-size 31 --algo word
```

#### Preprocessing

Then, we tokenize the texts with the built vocabulary and save the preprocessed samples.

```bash

```

#### Start training

- Different configurations for models in the paper are stored under the `configs/` directory. Launch a specific experiment with `--config` to specify the path to your desired model config and `--expname` to specify the name/number of this experiment which will be used in logging.
- We include three config files here: the baseline, KOBE without adding external knowledge, and full KOBE model.

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

- You can use TensorBoard. It can take (roughly) 12 hours for the training to stop. To get comparable results in paper, you need to train for even longer (by editing `epoch` in the config files). However, the current setting is enough to demonstrate the effectiveness of our model.

```bash
tensorboard --logdir experiments --port 6006
```

### Generation

- During training, the generated descriptions on the test set is saved at `experiments/<expname>/candidate.txt` and the ground truth is at `reference.txt`. This is generated by greedy search to save time in training and doesn't block repetitive terms.
- To do beam search with `beam width = 10`, run the following command.

```bash
python core/train.py --config configs/baseline.yaml --mode eval --restore experiments/finals-baseline/checkpoint.pt --expname eval-baseline --beam-size 10
```

### Evaluation

- BLEU
- DIVERSITY

If you have ANY difficulties to get things working in the above steps, feel free to open an issue.
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
