# Word Bigram Language Model

以繁體中文維基百科（2023）為語料，用 jieba 斷詞後建立詞層級的 bigram／trigram 語言模型，並做詞彙預測、句子機率、文字生成與 Zipf 定律等分析。

資訊檢索 HW2。

## 環境

- Python 3.14
- 安裝套件：

```bash
pip install -r requirements.txt
```

## 資料準備

以下檔案體積很大，未納入版本控制，需自行放到 `data/`：

| 檔案 | 說明 |
|---|---|
| `data/wiki_article_list_2023_tra.json` | 繁中維基文章列表（JSON 陣列，一行一篇，約 1.6 GB） |
| `data/dict.txt.big` | jieba 繁體加強詞典，可從 [jieba](https://github.com/fxsjy/jieba) 取得 |

## 執行流程

所有指令都在專案根目錄執行。

```bash
# 1. 把語料切成 100 個 shard → data/shards/raw/
python splitter.py

# 2. 斷句 + jieba 斷詞，每句前後加 <s> </s>（多行程）→ data/shards/tokens/
python tokenize_shards.py

# 3. 分別用前 10%、20%、…、100% 的 shard 計算 unigram / bigram → data/counts/
python count.py

# 4. 用 100% 語料計算 trigram → data/counts/tri_counts_p100.npz
python trigram/count_trigram.py
```

計數結果存成 `.npz`：bigram 以 `(w1 << 32) | w2` 編成一個 `uint64` key 並排序，查詢時用二分搜尋。

## 模型用法

```python
from bigram import BigramModel

m = BigramModel(K=100)                  # K = 使用的語料百分比
m.predict("臺灣", k=10)                  # 下一個詞的前 k 名與機率
m.prob("臺灣是一個美麗的島嶼。")            # (log P, 每一步的條件機率)
m.generate("臺灣", 20)                   # greedy 生成
m.generate("臺灣", 20, mode="sample")    # 依機率抽樣生成
m.entropy("的")                          # 下一個詞分布的 entropy（bits）
```

`trigram/trigram.py` 的 `TrigramModel` 介面相同。句子第一步，以及沒看過的兩詞歷史，會退回 bigram。

## 專案結構

```
├── splitter.py            語料切 shard
├── tokenize_shards.py     斷句、斷詞
├── count.py               unigram / bigram 計數
├── bigram.py              BigramModel
├── test.py                句子機率測試
├── trigram/
│   ├── count_trigram.py   trigram 計數
│   ├── trigram.py         TrigramModel
│   ├── compare.py         bigram vs trigram 生成與機率比較
│   └── test.py
├── experiments/
│   ├── Q1.py              greedy / sample 文字生成
│   ├── Q2.py              「的」「臺灣」「台灣」的後接詞分布與 entropy
│   ├── Q3.py              bigram 稀疏度 B / V²
│   ├── Q4.py              bigram 種類數 B 隨語料量 N 的成長
│   └── Q5.py              句子機率
├── zipf/zipf.py           unigram / bigram 的 Zipf 圖
├── chain/
│   ├── chain.py           從 <s> 開始生成句子：python chain/chain.py [bigram|trigram]
│   ├── chain_any.py       從任意詞開始生成句子
│   └── pick.py            互動式兩兩對決，挑出最通順／最好笑的句子
└── report/                報告（docx / pdf）與圖
```
