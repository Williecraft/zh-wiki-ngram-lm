import sys
sys.path.append(".")
sys.path.append("trigram")

import numpy as np

from bigram import BigramModel
from trigram import TrigramModel

name = sys.argv[1] if len(sys.argv) > 1 else "trigram"
if name == "bigram":
    tm = BigramModel(K=100)
    tm.bi = tm
else:
    tm = TrigramModel(K=100)

ids, c = tm.bi._followers("<s>")
picked = np.random.choice(ids, 128, replace=False, p=c / c.sum())

result = []
for i in picked:
    seed = tm.id2word[i]
    for _ in range(8):
        result.append(tm.generate(seed, 10**9, mode="sample"))

with open(f"chain/{name}/s/output.txt", 'w', encoding="utf8") as fout:
    fout.write("\n".join(result))

print(len(result))
