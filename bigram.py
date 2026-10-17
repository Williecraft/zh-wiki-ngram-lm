import math
from pathlib import Path

import numpy as np

from tokenize_shards import tokenize_article

COUNTS_PATH = Path("data/counts")
MASK = np.uint64(0xFFFFFFFF)


class BigramModel:
    def __init__(self, K=100):
        data = np.load(COUNTS_PATH / f"counts_p{K}.npz")
        self.unigram = data["unigram"]
        self.keys = data["keys"]
        self.counts = data["counts"]

        with open(COUNTS_PATH / f"vocab_p{K}.txt", 'r', encoding="utf8") as fin:
            self.id2word = fin.read().split("\n")
        self.word2id = {}
        for i, w in enumerate(self.id2word):
            self.word2id[w] = i

        self.V = len(self.id2word)

    def _get_id(self, w):
        if w not in self.word2id:
            raise KeyError(f"OOV: {w}")
        return self.word2id[w]

    def _followers(self, w):
        i = np.uint64(self._get_id(w))
        lo = np.searchsorted(self.keys, i << np.uint64(32))
        hi = np.searchsorted(self.keys, (i + np.uint64(1)) << np.uint64(32))
        ids = (self.keys[lo:hi] & MASK).astype(np.int64)
        return ids, self.counts[lo:hi]

    def _count(self, w1, w2=None):
        if w2 is None:
            if w1 not in self.word2id:
                return 0
            return int(self.unigram[self.word2id[w1]])
        if w1 not in self.word2id or w2 not in self.word2id:
            return 0
        key = (np.uint64(self.word2id[w1]) << np.uint64(32)) | np.uint64(self.word2id[w2])
        j = np.searchsorted(self.keys, key)
        if j < len(self.keys) and self.keys[j] == key:
            return int(self.counts[j])
        return 0

    def _history(self, w):
        if w not in self.word2id:
            return 0
        ids, c = self._followers(w)
        return int(c.sum())

    def _cond_prob(self, w2, w1):
        c12 = self._count(w1, w2)
        c1 = self._history(w1)
        if c1 == 0:
            return 0.0
        return c12 / c1

    def predict(self, w, k=10):
        ids, c = self._followers(w)
        total = c.sum()
        top = np.argsort(-c, kind="stable")[:k]
        result = []
        for j in top:
            result.append((self.id2word[ids[j]], float(c[j] / total)))
        return result

    def prob(self, sentence):
        if isinstance(sentence, str):
            words = []
            for w in tokenize_article(sentence).split():
                if w != "<s>" and w != "</s>":
                    words.append(w)
        else:
            words = list(sentence)

        log_p = 0.0
        steps = []
        for w1, w2 in zip(words[:-1], words[1:]):
            p = self._cond_prob(w2, w1)
            steps.append((w1, w2, p))
            if p == 0:
                log_p = -math.inf
            else:
                log_p += math.log(p)
        return log_p, steps

    def generate(self, seed, n=20, mode="greedy", sep='', stop=True):
        result = [seed]
        cur = seed
        while len(result) < n + 1:
            ids, c = self._followers(cur)
            if mode == "greedy":
                nxt = ids[np.argmax(c)]
            else:
                nxt = np.random.choice(ids, p=c / c.sum())
            cur = self.id2word[nxt]
            if cur == "</s>":
                if stop:
                    break
                cur = "<s>"
            else:
                result.append(cur)
        return sep.join(result)

    def entropy(self, w):
        ids, c = self._followers(w)
        p = c / c.sum()
        return float(-(p * np.log2(p)).sum()) + 0.0
