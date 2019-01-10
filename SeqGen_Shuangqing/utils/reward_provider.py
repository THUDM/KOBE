import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchtext import data
from torchtext import datasets
from utils.kc_dataset import KCDataset

# One of the key concepts in the FastText paper is that they calculate the n-grams of an input sentence and append them to the end of a sentence. Here, we'll use bi-grams. Briefly, a bi-gram is a pair of words/tokens that appear consecutively within a sentence.
#
# For example, in the sentence "how are you ?", the bi-grams are: "how are", "are you" and "you ?".
#
# The `generate_bigrams` function takes a sentence that has already been tokenized, calculates the bi-grams and appends them to the end of the tokenized list.


# def generate_bigrams(x):
#     n_grams = set(zip(*[x[i:] for i in range(2)]))
#     for n_gram in n_grams:
#         x.append(' '.join(n_gram))
#     return x


# def generate_trigrams(x):
#     n_grams = set(zip(*[x[i:] for i in range(3)]))
#     x = generate_bigrams(x)
#     for n_gram in n_grams:
#         x.append(' '.join(n_gram))
#     return x

# Build the Model
# class FastText(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, output_dim):
#         super().__init__()

#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.fc = nn.Linear(embedding_dim, output_dim)

#     def forward(self, x):
#         # x = [sent len, batch size]
#         embedded = self.embedding(x)

#         # embedded = [sent len, batch size, emb dim]
#         embedded = embedded.permute(1, 0, 2)
#         # embedded = [batch size, sent len, emb dim]

#         pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
#         # pooled = [batch size, embedding_dim]

#         return self.fc(pooled)


EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(
            fs, embedding_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # x = [sent len, batch size]
        x = x.permute(1, 0)
        # x = [batch size, sent len]

        embedded = self.embedding(x)
        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conv_n = [batch size, n_filters, sent len - filter_sizes[n]]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]
        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)


# We implement the function to calculate accuracy...
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


# We define a function for training our model...
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    tqdm_iterator = tqdm(iterator)
    for batch in tqdm_iterator:
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        tqdm_iterator.set_description(
            f"batch loss {loss:.4f}, batch acc {acc:.4f}")

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# We define a function for testing our model...
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def predict_sentiment(model, vocab, sentence):
    # User Input
    # And as before, we can test on any input the user provides.
    tokenized = sentence.split()
    if len(tokenized) < 5:
        tokenized += (5 - len(tokenized)) * [vocab.pad_token]
    indexed = [vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).cuda()
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()


class RewardProvider(object):

    def reward_fn(self, hypo, target):
        raise NotImplementedError

    def reward_fn_batched(self, hypos, targets):
        raise NotImplementedError


class CTRRewardProvider(RewardProvider):

    def __init__(self, path):
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage)
        self.vocab = checkpoint['vocab']
        input_dim = len(self.vocab)
        self.model = CNN(input_dim, EMBEDDING_DIM, N_FILTERS,
                         FILTER_SIZES, OUTPUT_DIM, DROPOUT)
        self.model.load_state_dict(checkpoint['model'])
        self.model.cuda()

    def reward_fn(self, hypo, target):
        hypo = ' '.join(hypo)
        target = ' '.join(target)
        return predict_sentiment(self.model, self.vocab, hypo)

    def reward_fn_batched(self, hypos, targets):
        max_len = max([len(tokenized) for tokenized in hypos])
        for tokenized in hypos:
            if len(tokenized) < max(max_len, 5):
                tokenized += (max(max_len, 5) - len(tokenized)) * ['<pad>']
        indexed_list = [[self.vocab.stoi[t] for t in tokenized] for tokenized in hypos]
        tensor = torch.LongTensor(indexed_list).t().cuda()
        prediction = torch.sigmoid(self.model(tensor)).squeeze()
        return prediction.tolist()


if __name__ == '__main__':

    SEED = 1234

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # TorchText `Field`s have a `preprocessing` argument. A function passed here will be applied to a sentence after it has been tokenized (transformed from a string into a list of tokens), but before it has been indexed (transformed from a list of tokens to a list of indexes). This is where we'll pass our `generate_bigrams` function.
    # TEXT = data.Field(tokenize=lambda x: x.split(),
    #                   preprocessing=generate_trigrams)
    TEXT = data.Field(tokenize=lambda x: x.split())
    LABEL = data.LabelField(dtype=torch.float)

    print('Loading dataset...')
    train_data, test_data = KCDataset.splits(TEXT, LABEL, root='../data')
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))

    # Build the vocab
    print('Building vocab...')
    TEXT.build_vocab(train_data)
    LABEL.build_vocab(train_data)

    # And create the iterators.
    BATCH_SIZE = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device)

    # As previously, we'll create an instance of our `FastText` class.

    INPUT_DIM = len(TEXT.vocab)

    # model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM)
    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS,
                FILTER_SIZES, OUTPUT_DIM, DROPOUT)

    # Train the Model
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    # Finally, we train our model...
    N_EPOCHS = 5
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(
            model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')

    # ...and get the test accuracy!
    #
    # The results are comparable to the results in the last notebook, but training takes considerably less time.
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')

    if not os.path.exists('experiments'):
        os.mkdir('experiments')
    SAVE_PATH = 'experiments/ctr_reward_provider.pt'
    torch.save({
        'model': model.state_dict(),
        'vocab': TEXT.vocab
    }, SAVE_PATH)
    print(f'CNN Model saved to {SAVE_PATH}')

    # An example negative review...
    print(predict_sentiment(model, TEXT.vocab, "小 白 鞋 夏 季 女 鞋 初 中 生 运 动 鞋 女 女 高 帮 鞋 <SEP> 运 动 鞋 休 闲 运 动 鞋 女 小 白 鞋 女 皮 面 学 生 孕 妇 鞋 透 气 女 学 生 豆 豆 鞋 单 鞋 运 动 鞋 女 厚 底 系 带 2 0 1 8 新 款 皮 女 单 鞋 透 气 护 士 鞋 网 布 鞋 女 网 面 休 闲 鞋 女 平 底 鞋 布 鞋 女 尖 头 鞋 女 平 底"))
    # An example positive review...
    print(predict_sentiment(model, TEXT.vocab, "这 些 超 火 的 i n s 网 红 公 仔 ， 萌 化 宝 宝 的 心 <SEP> 每 一 个 宝 宝 的 童 年 都 有 一 个 玩 具 公 仔 ， 其 实 这 些 公 仔 已 经 不 只 是 公 仔 ， 他 们 是 宝 宝 最 重 要 的 童 年 玩 伴 。 而 现 在 那 些 i n s 上 超 火 的 网 红 公 仔 ， 真 的 是 分 分 钟 萌 化 宝 宝 的 心 ， 你 给 宝 宝 准 备 好 了 么 ？"))
