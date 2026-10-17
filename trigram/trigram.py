import sys
sys.path.append(".")

import math
from pathlib import Path

import numpy as np

from bigram import BigramModel
from tokenize_shards import tokenize_article

COUNTS_PATH = Path("data/counts")
MASK = np.uint64(0xFFFFFFFF)


class TrigramModel:
    def __init__(self, K=100):
        data = np.load(COUNTS_PATH / f"tri_counts_p{K}.npz")
        self.unigram = data["unigram"]
        self.keys1 = data["keys1"]
        self.keys2 = data["keys2"]
        self.counts = data["counts"]

        with open(COUNTS_PATH / f"tri_vocab_p{K}.txt", 'r', encoding="utf8") as fin:
            self.id2word = fin.read().split("\n")
        self.word2id = {}
        for i, w in enumerate(self.id2word):
            self.word2id[w] = i

        self.V = len(self.id2word)
        self.bi = BigramModel(K)

    def _get_id(self, w):
        if w not in self.word2id:
            raise KeyError(f"OOV: {w}")
        return self.word2id[w]

    def _followers(self, w1, w2):
        i1 = np.uint64(self._get_id(w1))
        i2 = np.uint64(self._get_id(w2))
        lo = np.searchsorted(self.keys1, i1)
        hi = np.searchsorted(self.keys1, i1, side="right")
        sub = self.keys2[lo:hi]
        a = np.searchsorted(sub, i2 << np.uint64(32))
        b = np.searchsorted(sub, (i2 + np.uint64(1)) << np.uint64(32))
        ids = (sub[a:b] & MASK).astype(np.int64)
        return ids, self.counts[lo + a:lo + b]

    def _count(self, w1, w2=None, w3=None):
        if w3 is None:
            return self.bi._count(w1, w2)
        if w1 not in self.word2id or w2 not in self.word2id or w3 not in self.word2id:
            return 0
        ids, c = self._followers(w1, w2)
        return int(c[ids == self.word2id[w3]].sum())

    def _history(self, w1, w2):
        if w1 not in self.word2id or w2 not in self.word2id:
            return 0
        ids, c = self._followers(w1, w2)
        return int(c.sum())

    def _cond_prob(self, w3, w1, w2):
        c123 = self._count(w1, w2, w3)
        c12 = self._history(w1, w2)
        if c12 == 0:
            return 0.0
        return c123 / c12

    def predict(self, w1, w2, k=10):
        ids, c = self._followers(w1, w2)
        total = c.sum()
        top = np.argsort(-c, kind="stable")[:k]
        result = []
        for j in top:
            result.append((self.id2word[ids[j]], float(c[j] / total)))
        return result

    def prob(self, sentence):
        words = []
        for w in tokenize_article(sentence).split():
            if w != "<s>" and w != "</s>":
                words.append(w)

        log_p = 0.0
        steps = []
        for t in range(1, len(words)):
            w2 = words[t - 1]
            w3 = words[t]
            if t == 1:
                w1 = None
                p = self.bi._cond_prob(w3, w2)
            else:
                w1 = words[t - 2]
                p = self._cond_prob(w3, w1, w2)
            steps.append((w1, w2, w3, p))
            if p == 0:
                log_p = -math.inf
            else:
                log_p += math.log(p)
        return log_p, steps

    def generate(self, seed, n=20, mode="greedy", sep='', stop=True):
        result = [seed]
        prev = "<s>"
        cur = seed
        while len(result) < n + 1:
            if prev is None or self._history(prev, cur) == 0:
                ids, c = self.bi._followers(cur)
            else:
                ids, c = self._followers(prev, cur)
            if mode == "greedy":
                nxt = ids[np.argmax(c)]
            else:
                nxt = np.random.choice(ids, p=c / c.sum())
            w = self.id2word[nxt]
            if w == "</s>":
                if stop:
                    break
                prev = None
                cur = "<s>"
            else:
                result.append(w)
                prev = cur
                cur = w
        return sep.join(result)

    def entropy(self, w1, w2):
        ids, c = self._followers(w1, w2)
        p = c / c.sum()
        return float(-(p * np.log2(p)).sum()) + 0.0
