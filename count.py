import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

INPUT_PATH = Path("data/shards/tokens")
OUTPUT_PATH = Path("data/counts")
FILENAME = "shard{i}.txt"


def merge(keys, counts):
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    counts = counts[order]
    start = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    return keys[start], np.add.reduceat(counts, start)


for K in range(10, 101, 10):
    if (OUTPUT_PATH / f"counts_p{K}.npz").exists():
        continue

    start_time = time.perf_counter()

    word2id = {"<s>": 0, "</s>": 1}
    unigram = np.zeros(0, dtype=np.int64)
    all_keys = [np.zeros(0, dtype=np.uint64)]
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

        first = ids[:-1]
        second = ids[1:]
        ok = first != 1
        bigram = (first[ok] << np.uint64(32)) | second[ok]
        k, cnt = np.unique(bigram, return_counts=True)
        all_keys.append(k)
        all_counts.append(cnt)

        if (i + 1) % 10 == 0 or i == K - 1:
            keys, counts = merge(np.concatenate(all_keys), np.concatenate(all_counts))
            all_keys = [keys]
            all_counts = [counts]

    used_time = time.perf_counter() - start_time

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_PATH / f"counts_p{K}.npz", unigram=unigram, keys=keys, counts=counts)
    with open(OUTPUT_PATH / f"vocab_p{K}.txt", 'w', encoding="utf8") as fout:
        fout.write("\n".join(word2id))

    print(f"K = {K}%")
    print(f"N = {unigram.sum():,}（不含 <s> </s>：{unigram[2:].sum():,}）")
    print(f"V = {len(word2id):,}")
    print(f"B = {len(keys):,}")
    print(f"t = {used_time:.1f}s")
    print()
