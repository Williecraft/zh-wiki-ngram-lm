import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

K = 100

INPUT_PATH = Path("data/shards/tokens")
OUTPUT_PATH = Path("data/counts")
FILENAME = "shard{i}.txt"


def merge(keys1, keys2, counts):
    order = np.lexsort((keys2, keys1))
    keys1 = keys1[order]
    keys2 = keys2[order]
    counts = counts[order]
    start = np.flatnonzero(np.r_[True, (keys1[1:] != keys1[:-1]) | (keys2[1:] != keys2[:-1])])
    return keys1[start], keys2[start], np.add.reduceat(counts, start)


start_time = time.perf_counter()

word2id = {"<s>": 0, "</s>": 1}
unigram = np.zeros(0, dtype=np.int64)
all_keys1 = [np.zeros(0, dtype=np.uint64)]
all_keys2 = [np.zeros(0, dtype=np.uint64)]
all_counts = [np.zeros(0, dtype=np.int64)]

for i in tqdm(range(K), desc="Counting", unit=" shard", dynamic_ncols=True):
    with open(INPUT_PATH / FILENAME.format(i=i), 'r', encoding="utf8") as fin:
        words = fin.read().split()

    id_list = []
    for w in words:
        if w not in word2id:
            word2id[w] = len(word2id)
        id_list.append(word2id[w])
    ids = np.array(id_list, dtype=np.uint64)

    c = np.bincount(ids.astype(np.int64), minlength=len(word2id))
    unigram = np.pad(unigram, (0, len(c) - len(unigram))) + c

    first = ids[:-2]
    second = ids[1:-1]
    third = ids[2:]
    ok = (first != 1) & (second != 1)
    k1 = first[ok]
    k2 = (second[ok] << np.uint64(32)) | third[ok]
    k1, k2, cnt = merge(k1, k2, np.ones(len(k1), dtype=np.int64))
    all_keys1.append(k1)
    all_keys2.append(k2)
    all_counts.append(cnt)

    if (i + 1) % 10 == 0 or i == K - 1:
        keys1, keys2, counts = merge(np.concatenate(all_keys1), np.concatenate(all_keys2), np.concatenate(all_counts))
        all_keys1 = [keys1]
        all_keys2 = [keys2]
        all_counts = [counts]

used_time = time.perf_counter() - start_time

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
np.savez(OUTPUT_PATH / f"tri_counts_p{K}.npz", unigram=unigram, keys1=keys1, keys2=keys2, counts=counts)
with open(OUTPUT_PATH / f"tri_vocab_p{K}.txt", 'w', encoding="utf8") as fout:
    fout.write("\n".join(word2id))

print(f"K = {K}%")
print(f"N = {unigram.sum():,}（不含 <s> </s>：{unigram[2:].sum():,}）")
print(f"V = {len(word2id):,}")
print(f"T = {len(keys1):,}")
print(f"t = {used_time:.1f}s")
