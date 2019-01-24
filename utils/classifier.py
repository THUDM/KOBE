import os
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchtext import data
from utils.aspect_dataset import AspectDataset


EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 40
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

def compute_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    pred_y = torch.argmax(preds, dim=1)
    correct = (pred_y == y).float()  # convert into float for division
    acc = correct.mean()
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
        acc = compute_accuracy(predictions, batch.label)

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
            acc = compute_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def predict(model, vocab, sentence):
    # User Input
    # And as before, we can test on any input the user provides.
    tokenized = sentence.split()
    if len(tokenized) < 5:
        tokenized += (5 - len(tokenized)) * [vocab.pad_token]
    indexed = [vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).cuda()
    tensor = tensor.unsqueeze(1)
    prediction = F.softmax(model(tensor))
    return prediction

def predict_batched(model, vocab, sentences):
    hypos = [sentence.split() for sentence in sentences]
    max_len = max([len(tokenized) for tokenized in hypos])
    for tokenized in hypos:
        if len(tokenized) < max(max_len, 5):
            tokenized += (max(max_len, 5) - len(tokenized)) * ['<pad>']
    indexed_list = [[vocab.stoi[t] for t in tokenized] for tokenized in hypos]
    tensor = torch.LongTensor(indexed_list).t().cuda()
    prediction = F.softmax(model(tensor))
    return prediction


def transform_dataset(fname_src, fname_tgt, dirname):
    f_src = open(fname_src)
    f_tgt = open(fname_tgt)
    labels = [line.split()[0] for line in f_src.read().strip().split('\n')]
    words = f_tgt.read().strip().split('\n')
    assert len(labels) == len(words)
    output = [word + ' \t ' + label for word, label in zip(words, labels)]
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    with open(os.path.join(dirname, 'dataset.txt'), 'w') as f:
        f.write('\n'.join(output))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('train_src', type=str)
    parser.add_argument('train_tgt', type=str)
    args = parser.parse_args()
    transform_dataset(args.train_src, args.train_tgt, 'train')

    SEED = 1234

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # TorchText `Field`s have a `preprocessing` argument. A function passed here will be applied to a sentence after it has been tokenized (transformed from a string into a list of tokens), but before it has been indexed (transformed from a list of tokens to a list of indexes). This is where we'll pass our `generate_bigrams` function.
    # TEXT = data.Field(tokenize=lambda x: x.split(),
    #                   preprocessing=generate_trigrams)
    TEXT = data.Field(tokenize=lambda x: x.split())
    LABEL = data.LabelField(dtype=torch.long)

    print('Loading dataset...')
    train_data, = AspectDataset.splits(TEXT, LABEL, test=None, root='.')
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))

    # Build the vocab
    print('Building vocab...')
    TEXT.build_vocab(train_data)
    LABEL.build_vocab(train_data)

    # And create the iterators.
    BATCH_SIZE = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_data, valid_data),
        batch_size=BATCH_SIZE,
        device=device)

    # As previously, we'll create an instance of our `FastText` class.
    vocab = TEXT.vocab
    INPUT_DIM = len(vocab)

    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS,
                FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)
    # Finally, we train our model...
    N_EPOCHS = 5
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(
            model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')

    # Save the model
    SAVE_PATH = 'checkpoint.pt'
    torch.save({
        'model': model.state_dict(),
        'vocab': TEXT.vocab,
        'tag_vocab': LABEL.vocab
    }, SAVE_PATH)
    print(f'CNN Model saved to {SAVE_PATH}')
