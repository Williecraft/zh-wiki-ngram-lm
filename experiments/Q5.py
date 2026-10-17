import sys
sys.path.append(".")

from bigram import BigramModel

m = BigramModel(K=100)

sentences = [
    "今天天氣很好。",
    "我喜歡吃蘋果。",
    "臺灣是一個美麗的島嶼。",
    "這部電影在美國上映後獲得好評。",
    "他畢業於國立臺灣大學。",
    "蘋果吃喜歡我。",
]

for s in sentences:
    log_p, steps = m.prob(s)
    print(s, log_p)
    for w1, w2, p in steps:
        print("\t", w1, w2, p)
    print()
