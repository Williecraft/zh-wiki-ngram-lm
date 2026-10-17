from trigram import TrigramModel

tm = TrigramModel(K=100)
bm = tm.bi

print("B", len(bm.keys))
print("T", len(tm.keys1))
print()

for seed in ["臺灣", "我", "數學", "電影"]:
    print(seed)
    print("bigram  greedy", bm.generate(seed, 30))
    print("trigram greedy", tm.generate(seed, 30))
    print("bigram  sample", bm.generate(seed, 30, mode="sample"))
    print("trigram sample", tm.generate(seed, 30, mode="sample"))
    print()

for s in ["我喜歡吃蘋果。", "臺灣是一個美麗的島嶼。", "他畢業於國立臺灣大學。"]:
    print(s, bm.prob(s)[0], tm.prob(s)[0])
