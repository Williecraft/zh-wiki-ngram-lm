import numpy as np

data = np.load("data/counts/counts_p100.npz")
V = len(data["unigram"])
B = len(data["keys"])

print("V", V)
print("B", B)
print("B / V^2", B / (V * V))
