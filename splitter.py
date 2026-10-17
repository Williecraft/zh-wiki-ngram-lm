import json
from pathlib import Path
from tqdm import tqdm

INPUT_FILE = "data/wiki_article_list_2023_tra.json"
OUTPUT_PATH = Path("data/shards/raw")
FILENAME = "shard{i}.txt"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

fouts = []
for i in range(100):
    p = OUTPUT_PATH / FILENAME.format(i=i)
    fouts.append(open(p, 'w', encoding="utf8"))

with open(INPUT_FILE, 'r', encoding="utf8") as fin:
    n = 0
    for line in tqdm(fin, total=1_323_056, desc="Splitting", unit=" lines", unit_scale=True, dynamic_ncols=True):
        line = line.strip()
        if line in ("[", "]"): continue

        line:str = json.loads(line.rstrip(','))
        fouts[n%100].write(line.replace('\n', ' ') + '\n')

        n += 1

for i in fouts:
    i.close()
