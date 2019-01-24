import pickle as pkl
import numpy as np
import thulac
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy import spatial
from gensim.models import Word2Vec
from utils.clean import get_chinese
from utils.classifier import CNN, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT


class Tagger(object):
    def tag(self, sentence):
        raise NotImplementedError


class AspectTagger(Tagger):
    def __init__(self):
        self.ASPECTS_KEYWORDS = {
            'appearance': [
                '简约_a', '美观_a', '优雅_a', '美_a'
            ],
            'texture': [
                '柔软_a', '细腻_a', '天然_a', '舒服_a'
            ],
            'function': [
                '方便_a', '耐用_a', '健康_a', '结实_a'
            ],
        }
        self._load_aspect_embeddings()
        self.tokenizer = thulac.thulac()

    def _load_aspect_embeddings(self):
        # Pretrained word embeddings on segmented dataset
        self.model = Word2Vec.load('data/emotion/word2vec.model')
        # Pre-computed word freqs
        counter = pkl.load(open('data/emotion/counter.pkl', 'rb'))
        adj_cnt_list = [(counter[w], w) for w in counter if w.endswith('_a')]

        adj_cnt_list = list(reversed(sorted(adj_cnt_list)))

        NUM_LABELED = 5000

        score = np.zeros((NUM_LABELED, len(self.ASPECTS_KEYWORDS)))
        for idx_w, (_, w) in enumerate(adj_cnt_list[:NUM_LABELED]):
            for idx_asp, aspect in enumerate(self.ASPECTS_KEYWORDS):
                score[idx_w, idx_asp] = max(
                    map(lambda x: self.model.similarity(
                        x, w), self.ASPECTS_KEYWORDS[aspect])
                )

        self.aspect_collections = [[]
                                   for _ in range(len(self.ASPECTS_KEYWORDS))]

        for idx_w, (_, w) in enumerate(adj_cnt_list[:NUM_LABELED]):
            if np.max(score[idx_w]) / np.sum(score[idx_w]) > 0.8:
                self.aspect_collections[np.argmax(score[idx_w])].append(w)

        self.aspect_embeddings = [np.mean([self.model.wv[w] for w in collection], axis=0)
                                  for collection in self.aspect_collections]

    def tag(self, sentence, tagname=False, require_scores=False):
        """Tokenize, embed and finally tag the sentence."""
        sentence = get_chinese(sentence)
        tokenized = self.tokenizer.cut(sentence, text=True).split()
        embeded = np.mean([self.model.wv[w]
                           for w in tokenized if w in self.model.wv], axis=0)
        tag_id = np.argmin([spatial.distance.cosine(embeded, aspect_embedding)
                            for aspect_embedding in self.aspect_embeddings])
        if require_scores:
            scores = [1 - spatial.distance.cosine(self.model.wv[w], self.aspect_embeddings[2])
                      for w in tokenized if w in self.model.wv]
            tokenized = [w[:w.index('_')] for w in tokenized if w in self.model.wv]
            return tag_id, (scores, tokenized)
        return tag_id if not tagname else (tag_id, list(self.ASPECTS_KEYWORDS.keys())[tag_id])


class InterestTagger(Tagger):
    def __init__(self):
        checkpoint = torch.load(
            'data/checkpoint.pt', map_location=lambda storage, loc: storage)
        self.vocab = checkpoint['vocab']
        self.tag_vocab = checkpoint['tag_vocab']
        self.model = CNN(len(self.vocab), EMBEDDING_DIM, N_FILTERS,
                         FILTER_SIZES, OUTPUT_DIM, DROPOUT)
        self.model.cuda()
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.criterion = nn.CrossEntropyLoss()

    def tag(self, sentence, tagname=False, require_scores=False):
        sentence = get_chinese(sentence)
        tokenized = list(sentence)
        if len(tokenized) < 5:
            tokenized += (5 - len(tokenized)) * ['<pad>']
        indexed = [self.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).cuda()
        tensor = tensor.unsqueeze(1)

        # with torch.no_grad():
        output = self.model(tensor)
        prediction = F.softmax(output)
        tag_id = prediction.argmax()

        # output logits
        x = tensor.permute(1, 0)
        embedded = self.model.embedding(x)
        embedded = embedded.unsqueeze(1)
        embedded.retain_grad()
        conved = [F.relu(conv(embedded)).squeeze(3)
                  for conv in self.model.convs]
        # pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
        #           for conv in conved]
        # logits = torch.cat(pooled, dim=1)
        # prediction = self.model.fc(logits)
        logits = conved[0].squeeze()
        w = list(self.model.fc.parameters())[0].data[tag_id][:100]
        scores = torch.matmul(w, logits)
        scores = ((torch.max(scores, torch.Tensor([0]).expand_as(
            scores).cuda())) / scores.max() * 50).tolist()
        # self.model.zero_grad()
        # loss = self.criterion(prediction, torch.LongTensor([0]).cuda())
        # loss = self.criterion(prediction, tag_id[None].detach())
        # loss.backward()
        # scores = embedded.grad.detach().squeeze()
        # scores = scores.norm(dim=1)
        # scores = ((scores - scores.min()) / (scores.max() - scores.min()) * 50).tolist()
        # print(list(self.model.embedding.parameters()[0].grad))
        # print(logits.grad)
        # print(embedded.grad.shape)

        tag_id = tag_id.item()
        if require_scores:
            return tag_id, scores
        return tag_id if not tagname else (tag_id, self.tag_vocab.itos[tag_id])


def scored_sentence(sentence, scores, color="teal"):
    output_string = ""
    scores = np.array(scores)
    scores = (scores - scores.min()) / (scores.max() - scores.min()) * 50
    for char, score in zip(sentence, scores):
        if char in "！？，。；、,.!?;":
            score = 0
        output_string += f"\\cjkhl{{{color}!{int(score)}}}{{{char}}} "
    return output_string


if __name__ == "__main__":
    tagger = AspectTagger()
    # print(tagger.tag(
        # "运动鞋休闲运动鞋女小白鞋女皮面学生孕妇鞋透气女学生豆豆鞋单鞋运动鞋女厚底系带2018新款皮女单鞋透气护士鞋网布鞋女网面休闲鞋女平底鞋布鞋女尖头鞋女平底"))
    # print(tagger.tag("出国转换器，全球通多用插头，去不同国家履行，解决不同电器的插头，随意组合，独立使用", tagname=True))
    tag_id, (scores, tokenized) = tagger.tag("采用天然柳条编织而成，纯手工编织，纹理清晰，色泽自然，散发着自然清新的自然气息，带着淡淡的清香，让人心旷神怡。", require_scores=True)
    print(scored_sentence(tokenized, scores, color='brown'))
    tag_id, (scores, tokenized) = tagger.tag("采用天然柳条编织而成，纯手工编织，结实耐用，不易变形，经久耐用。可折叠的设计，方便收纳，携带更方便。", require_scores=True)
    print(scored_sentence(tokenized, scores, color='brown'))
    # interest_tagger = InterestTagger()
    # sentence = "运动鞋休闲运动鞋女小白鞋女皮面学生孕妇鞋透气女学生豆豆鞋单鞋运动鞋女厚底系带2018新款皮女单鞋透气护士鞋网布鞋女网面休闲鞋女平底鞋布鞋女尖头鞋女平底"
    # tag, scores = interest_tagger.tag(sentence, require_scores=True)
    # print(scored_sentence(sentence[1:], scores))
    # sentence = "出国转换器，全球通多用插头，去不同国家旅行，解决不同电器的插头，随意组合，独立使用"
    # tag, scores = interest_tagger.tag(sentence, require_scores=True)
    # print(scored_sentence(sentence[1:], scores))
    # sentence = "这款鞋子采用了优质的头层牛皮，质感细 腻，柔软舒适，透气性好，鞋底采用了橡 胶材质，防滑耐磨。"
    # tag, scores = interest_tagger.tag(sentence, require_scores=True)
    # print(tag, scored_sentence(sentence[1:], scores))
    # sentence = "经典的黑 白撞色拼接，时尚又个性，搭配上黑色的 牛仔裤，更显时尚潮流。"
    # tag, scores = interest_tagger.tag(sentence, require_scores=True)
    # print(tag, scored_sentence(sentence[1:], scores))
