import sys
sys.path.append(".")

from bigram import BigramModel

m = BigramModel(K=100)

for seed in ["臺灣", "我", "數學", "電影", "大學", "他", "音樂", "公司", "歷史", "我們"]:
    print(seed)
    print("greedy", m.generate(seed, 40))
    for _ in range(2):
        print("sample", m.generate(seed, 40, mode="sample"))
    print()
