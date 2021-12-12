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
- [Experiments](#experiments)
  - [Visualization with WandB](#visualization-with-wandb)
  - [Training your own KOBE](#training-your-own-kobe)
  - [Testing KOBE](#testing-kobe)
- [Preprocessing](#preprocessing)
  - [Build vocabulary](#build-vocabulary)
  - [Preprocessing](#preprocessing-1)
- [Cite](#cite)

## Prerequisites

- Linux
- Python >= 3.8
- PyTorch >= 1.10

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

## Experiments

### Visualization with WandB

First, set up [WandB](https://wandb.ai/), which is an 🌟 incredible tool for visualize deep learning experiments. In case you haven't use it before, please login and follow the instructions.

```bash
wandb login
```

### Training your own KOBE

We provide four training modes here: `baseline`, `kobe-attr`, `kobe-know`, `kobe-full`, corresponding to the models explored in the paper. They can be trained with the following commands:

```bash
python -m kobe.train --mode baseline --name baseline
python -m kobe.train --mode kobe-attr --name kobe-attr
python -m kobe.train --mode kobe-know --name kobe-know
python -m kobe.train --mode kobe-full --name kobe-full
```

After launching any of the experiment above, please go to the WandB link printed out in the terminal to view the training/validation loss, BLEU, and even the generated examples (updated once every epoch) there!

If you would like to change other hyperparameters, please look at `kobe/utils/options.py`.

### Testing KOBE

TODO

## Preprocessing

Preprocessing is a commonly neglected part in code release. However, we now provide the preprocessing scripts to rebuild the vocabulary and tokenize the texts, just in case that you wish to preprocess the KOBE data yourself or need to run on your own data.

### Build vocabulary

We use BPE to build a vocabulary on the texts and conditions (including attributes and user categories).

```bash
python -m kobe.data.vocab \
  --input data-v2/raw/train.title data-v2/raw/train.desc data-v2/raw/train.fact \
  --vocab-file data-v2/vocab.text \
python -m kobe.data.vocab \
  --input data-v2/raw/train.cond \
  --vocab-file data-v2/vocab.cond \
  --vocab-size 31 --algo word
```

### Preprocessing

Then, we tokenize the inputs with the built vocabulary and save the preprocessed samples to `.tar` files.

```bash
python -m kobe.data.preprocess \
  --raw-path data-v2/raw/ \
  --processed-path data-v2/processed/ \
  --split train valid test \
  --vocab-file data-v2/vocab.text.model \
  --cond-vocab-file data-v2/vocab.cond.model
```

You can peek into the `data-v2/raw/` and `data-v2/processed/` directories to see what these scripts do!

## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{chen2019towards,
  title={Towards knowledge-based personalized product description generation in e-commerce},
  author={Chen, Qibin and Lin, Junyang and Zhang, Yichang and Yang, Hongxia and Zhou, Jingren and Tang, Jie},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={3040--3050},
  year={2019}
}
```
