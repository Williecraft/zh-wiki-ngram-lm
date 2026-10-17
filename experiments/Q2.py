import sys
sys.path.append(".")

from bigram import BigramModel

m = BigramModel(K=100)

for w in ["的", "臺灣", "台灣"]:
    print(w, m.unigram[m.word2id[w]], len(m.predict(w, m.V)), m.entropy(w))
    for word, p in m.predict(w, 10):
        print("\t", word, p)
    print()
