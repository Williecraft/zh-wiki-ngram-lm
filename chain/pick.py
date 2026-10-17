import random


def ask(prompt, options):
    while True:
        ans = input(prompt).strip()
        if ans in options:
            return options[ans]


kind = ask("1. 通順  2. 好笑：", {"1": "fluent", "2": "funny"})
model = ask("1. bigram  2. trigram：", {"1": "bigram", "2": "trigram"})
start = ask("1. <s>  2. 任意：", {"1": "s", "2": "any"})

with open(f"chain/{model}/{start}/{kind}.txt", encoding="utf8") as fin:
    alive = [line.strip() for line in fin if line.strip()]
random.shuffle(alive)
level = {s: 0 for s in alive}
seen = {s: -1 for s in alive}
total = len(alive) - 2

for t in range(total):
    best = None
    for lv in set(level[s] for s in alive):
        group = sorted([s for s in alive if level[s] == lv], key=lambda s: seen[s])
        if len(group) >= 2 and (best is None or seen[group[1]] < seen[best[1]]):
            best = group[:2]
    if best is None:
        break
    a, b = best
    print(f"\n[第 {t + 1}/{total} 場，{(total + 2) >> level[a]} 強]")
    print("1.", a)
    print("2.", b)
    win = ask("選 1 或 2：", {"1": a, "2": b})
    alive.remove(b if win == a else a)
    level[win] += 1
    seen[a] = seen[b] = t

with open(f"chain/{model}/{start}/{kind}_top2.txt", 'w', encoding="utf8") as fout:
    fout.write("\n".join(alive))

print("\n2 強：")
print("\n".join(alive))
