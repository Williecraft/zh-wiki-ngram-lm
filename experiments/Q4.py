import numpy as np
import matplotlib.pyplot as plt

Ns = []
Bs = []
for K in [1] + list(range(10, 101, 10)):
    data = np.load(f"data/counts/counts_p{K}.npz")
    N = int(data["unigram"].sum())
    V = len(data["unigram"])
    B = len(data["keys"])
    Ns.append(N)
    Bs.append(B)
    print(K, N, V, B)

plt.loglog(Ns, Bs, "o-")
plt.xlabel("N")
plt.ylabel("B")
plt.savefig("experiments/Q4.png")
