import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import re
import logging
from multiprocessing import Pool
from pathlib import Path

import jieba
from tqdm import tqdm

INPUT_PATH = Path("data/shards/raw")
OUTPUT_PATH = Path("data/shards/tokens")
FILENAME = "shard{i}.txt"
SENT_END = re.compile(r"(?<=[。！？!?；;])")

jieba.setLogLevel(logging.WARNING)
jieba.set_dictionary("data/dict.txt.big")


def tokenize_article(text):
    result = []
    for sentence in SENT_END.split(text):
        words = []
        for w in jieba.cut(sentence):
            if not w.isspace():
                words.append(w)
        if len(words) > 0:
            result.append("<s> " + " ".join(words) + " </s>")
    return " ".join(result)


def tokenize_shard(i):
    fin = open(INPUT_PATH / FILENAME.format(i=i), encoding="utf8")
    fout = open(OUTPUT_PATH / FILENAME.format(i=i), "w", encoding="utf8")
    for line in fin:
        line = line.rstrip("\n")
        fout.write(tokenize_article(line) + "\n")
    fin.close()
    fout.close()


if __name__ == "__main__":
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    pool = Pool()
    for _ in tqdm(pool.imap_unordered(tokenize_shard, range(100)), total=100, desc="Tokenizing", unit=" shard", dynamic_ncols=True):
        pass
    pool.close()
    pool.join()
