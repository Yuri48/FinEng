"""Step 2a: document-level sentiment, three ways.

1. tone_rob   cardiffnlp/twitter-roberta-base-sentiment-latest (RoBERTa fine-tuned on ~124M tweets + TweetEval):
              P(positive) - P(negative), in [-1, 1]. Social-media register, works on headlines too.
2. tone_vader VADER compound score, the lexicon baseline.
3. pro_dem    Partisan direction. Overall tone does not say who is being criticised ("Trump's tariffs are crushing
              farmers" and "Democrats are crushing it" have opposite tone and the same partisan meaning), so for every
              document that names a party or its leaders, an aspect-based sentiment model
              (yangheng/deberta-v3-base-absa-v1.1) scores the sentiment *toward* the first-named Republican-side
              entity and the first-named Democratic-side entity. pro_dem = mean of (+tone toward Democrats,
              -tone toward Republicans) over the sides mentioned; positive = the document favours Democrats.

Scores are cached by text (md5), so reruns only score new documents. -> data/doc_sentiment.parquet
"""
import hashlib, json, re
import numpy as np
import pandas as pd
from config import DATA
from corpus import load_news

ROB = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ABSA = "yangheng/deberta-v3-base-absa-v1.1"
GOP = re.compile(r"\b(Trump|Republicans?|GOP|MAGA|JD Vance|Vance|Mike Johnson|Speaker Johnson|Thune|Hegseth|Noem|Bondi|"
                 r"Stephen Miller|Leavitt|RFK Jr|Musk)\b")
DEM = re.compile(r"\b(Democrats?|Democratic Party|Dems?|DNC|Schumer|Jeffries|Kamala Harris|Harris|Newsom|Pelosi|"
                 r"Ocasio-Cortez|AOC|Obama|Ken Martin|Mamdani)\b")
ISSUE_OF_QUERY = {"economy": "Economy & cost of living", "democracy": "Democracy & election integrity",
                  "iran_war": "Iran war & foreign policy", "corruption": "Corruption & accountability",
                  "immigration": "Immigration & ICE", "health_care": "Health care", "tariffs": "Tariffs & trade",
                  "ai_tech": "AI, tech & data centres"}


def md5(t): return hashlib.md5(t.encode()).hexdigest()


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _cache(name):
    f = DATA / f"cache_{name}.parquet"
    return (pd.read_parquet(f) if f.exists() else pd.DataFrame()), f


def roberta(texts, batch=64):
    cache, f = _cache("roberta")
    have = set(cache.index) if len(cache) else set()
    todo = list(dict.fromkeys(t for t in texts if md5(t) not in have))
    if todo:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(ROB); m = AutoModelForSequenceClassification.from_pretrained(ROB).to(_device()).eval()
        lab = [m.config.id2label[i].lower() for i in range(3)]
        rows = []
        with torch.no_grad():
            for i in range(0, len(todo), batch):
                b = todo[i:i + batch]
                enc = tok(b, padding=True, truncation=True, max_length=160, return_tensors="pt").to(m.device)
                p = torch.softmax(m(**enc).logits, -1).cpu().numpy()
                rows += [dict(zip(lab, r)) for r in p]
                if (i // batch) % 100 == 0: print("roberta", i, "/", len(todo), flush=True)
        new = pd.DataFrame(rows, index=[md5(t) for t in todo])
        cache = pd.concat([cache, new]); cache.to_parquet(f)
    c = cache.loc[[md5(t) for t in texts]]
    return (c["positive"] - c["negative"]).to_numpy(), c["negative"].to_numpy(), c["positive"].to_numpy()


def vader(texts):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    a = SentimentIntensityAnalyzer()
    return np.array([a.polarity_scores(t)["compound"] for t in texts])


def absa(pairs, batch=24, save_every=40):
    """pairs: list of (text, aspect). Returns P(pos) - P(neg) toward the aspect.
    Sorted by length (less padding), small batches, MPS cache released and progress saved every `save_every` batches:
    PyTorch's MPS allocator otherwise keeps a buffer per padded shape and a long run swaps a 16 GB machine."""
    cache, f = _cache("absa")
    key = lambda t, a: md5(a + "||" + t)
    have = set(cache.index) if len(cache) else set()
    todo = sorted(dict.fromkeys((t, a) for t, a in pairs if key(t, a) not in have), key=lambda x: len(x[0]))
    if todo:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(ABSA); m = AutoModelForSequenceClassification.from_pretrained(ABSA).to(_device()).eval()
        lab = [m.config.id2label[i].lower() for i in range(m.config.num_labels)]
        out, keys = [], []
        with torch.no_grad():
            for j, i in enumerate(range(0, len(todo), batch)):
                b = todo[i:i + batch]
                enc = tok([t for t, _ in b], [a for _, a in b], padding=True, truncation="only_first", max_length=160,
                          return_tensors="pt").to(m.device)
                p = torch.softmax(m(**enc).logits, -1).cpu().numpy()
                out += [dict(zip(lab, r)) for r in p]; keys += [key(t, a) for t, a in b]
                del enc
                if (j + 1) % save_every == 0 or i + batch >= len(todo):
                    cache = pd.concat([cache, pd.DataFrame(out, index=keys)]); cache.to_parquet(f); out, keys = [], []
                    if m.device.type == "mps": torch.mps.empty_cache()
                    print("absa", min(i + batch, len(todo)), "/", len(todo), flush=True)
        del m
    cache = cache[~cache.index.duplicated(keep="last")]
    c = cache.loc[[key(t, a) for t, a in pairs]]
    return (c["positive"] - c["negative"]).to_numpy()


def partisan(df):
    g = df.text.str.extract(GOP, expand=False); d = df.text.str.extract(DEM, expand=False)
    df["gop_aspect"], df["dem_aspect"] = g, d
    df["tone_gop"], df["tone_dem"] = np.nan, np.nan
    for side, col in (("gop_aspect", "tone_gop"), ("dem_aspect", "tone_dem")):
        m = df[side].notna()
        df.loc[m, col] = absa(list(zip(df.loc[m, "text"], df.loc[m, side])))
    df["pro_dem"] = pd.concat([df.tone_dem, -df.tone_gop], axis=1).mean(axis=1)    # NaN if neither side named
    return df


def stage2_news():
    """Issue-anchored headlines from collect_news.py topics; the issue is the query's."""
    files = sorted(DATA.glob("news_topics*.jsonl"))                 # the stage-2 run was sharded by query
    if not files: return pd.DataFrame()
    merged = DATA / "_news_topics_merged.jsonl"
    with open(merged, "w") as out:
        for f in files: out.write(open(f).read())
    n = load_news(merged, tag="news2").drop_duplicates(["q", "text"])
    n["issue_q"] = n.q.map(ISSUE_OF_QUERY)
    n["week"] = pd.to_datetime(n.date).dt.to_period("W-SUN").dt.end_time.dt.normalize().dt.date
    return n


if __name__ == "__main__":
    corp = pd.read_parquet(DATA / "corpus.parquet")
    s2 = stage2_news()
    cols = ["doc_id", "source", "kind", "date", "week", "text", "weight", "lean", "community"]
    df = pd.concat([corp[cols + ["topic"]], s2[cols + ["issue_q"]]], ignore_index=True)
    print(df.source.value_counts().to_dict())
    df["tone_rob"], df["p_neg"], df["p_pos"] = roberta(df.text.tolist())
    df["tone_vader"] = vader(df.text.tolist())
    df = partisan(df)
    df.drop(columns=["text"]).to_parquet(DATA / "doc_sentiment.parquet")
    print(df.groupby("source")[["tone_rob", "tone_vader", "pro_dem"]].agg(["mean", "count"]).round(3))
