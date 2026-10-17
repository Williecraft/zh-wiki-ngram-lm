from bigram import BigramModel

m = BigramModel(K=100)

# print("===== predict =====")
# for w in ["的", "臺灣", "台灣"]:
#     print(w)
#     for word, p in m.predict(w, 5):
#         print(f"\t{word}\t{p:.4f}")

# print()
print("===== prob =====")
sentences = [
    "臺灣是一個美麗的島嶼。",
    "臺灣島嶼是美麗的。",
    "今天天氣真好。"
]
for s in sentences:
    log_p, steps = m.prob(s)
    print(s)
    for t in steps:
        print(t[0], end=" ")
    print(steps[-1][1]) 
    print(f"log P = {log_p:.2f}\n")

# print()
# print("===== generate =====")
# for seed in ["臺灣", "我"]:
#     print(seed, "greedy:", m.generate(seed, 20))
#     for _ in range(2):
#         print(seed, "sample:", m.generate(seed, 20, mode="sample"))

# print()
# print("===== entropy =====")
# for w in ["的", "臺灣"]:
#     print(f"{w}\t{m.entropy(w):.2f} bits")
