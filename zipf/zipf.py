import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/counts/counts_p100.npz")
unigram = np.sort(data["unigram"][2:])[::-1]
bigram = np.sort(data["counts"])[::-1]

plt.loglog(np.arange(1, len(unigram) + 1), unigram, label="unigram")
plt.loglog(np.arange(1, len(bigram) + 1), bigram, label="bigram")
plt.xlabel("rank")
plt.ylabel("frequency")
plt.legend()
plt.savefig("zipf/zipf.png")
