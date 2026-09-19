"""
NLP construction of the daily 'Iran war news intensity' features used to select the high-variance (H) days.
Corpora: (1) Google News RSS headlines (two daily queries), (2) Wikipedia Portal:Current events daily pages,
         (3) Wikipedia '2026 Iran war' timeline daily sections, (4) GDELT volume/tone (if the API allowed the pull).
Steps: relevance filter (regex) -> escalation lexicon score -> FinBERT tone -> TF-IDF novelty -> volume features.
"""
import compat  # noqa: F401  (scipy PROPACK stub, see compat.py)
import json, re, os, sys, math, numpy as np, pandas as pd
from collections import Counter

WAR_RE = re.compile(r"\b(?:iran|iranian|tehran|irgc|khamenei|hormuz|persian gulf|gulf states?|houthi|bab[ -]el[ -]mandeb|islamabad (?:talks|memorandum)|epic fury|araghchi|pezeshkian|mojtaba)\b", re.I)
CONFLICT_RE = re.compile(r"\b(?:war|strike[sd]?|attack(?:s|ed)?|missile[s]?|drone[s]?|bomb(?:s|ing|ed)?|blockade|ceasefire|cease-fire|truce|talks|negotiat\w*|deal|memorandum|tanker[s]?|hormuz|escalat\w*|retaliat\w*|nuclear|ultimatum|deadline|sanction[s]?|oil|troops?|carrier|military|killed|casualt\w*|peace|surrender|regime|iaea|uranium|enrich\w*)\b", re.I)
ESC_POS = ["strike","strikes","struck","attack","attacks","attacked","missile","missiles","drone","drones","bomb","bombs","bombing","bombed","blockade","closes","closed","closure","kill","killed","killing","escalat","retaliat","retaliation","ultimatum","deadline","threat","threatens","threatened","mine","mines","mined","sinks","sunk","sank","hit","hits","explosion","war","warship","carrier","troops","deploy","deployment","surrender","dead","casualties","collapse","collapses","expires","expired","seize","seized","hijack","fire","fires","fired","offensive","barrage","resumes","resumed","invasion","nuclear","enrichment","warheads"]
ESC_NEG = ["ceasefire","cease-fire","truce","deal","talks","agreement","agree","agrees","agreed","memorandum","reopen","reopens","reopened","de-escalat","deescalat","pause","paused","negotiat","negotiation","negotiations","peace","withdraw","withdrawal","signed","signing","signs","stand-down","halt","halts","halted","calm","relief","diplomacy","diplomatic","mediat","extend","extends","extended","lifts","lifted","open","opens","corridor","inspection","inspections","visa","meeting","summit","envoy"]

def esc_score(t):
    tl = t.lower(); p = sum(tl.count(w) for w in ESC_POS); n = sum(tl.count(w) for w in ESC_NEG)
    return ((p - n) / (p + n) if (p + n) else 0.0), p + n

def load_gnews(path="data/gnews_headlines.jsonl"):
    rows = []
    for l in open(path):
        j = json.loads(l)
        for it in j["items"]:
            t = re.sub(r"\s+-\s+[^-]+$", "", it.get("title") or "")
            rows.append({"day": j["day"], "q": j["q"], "title": t, "source": it.get("source") or ""})
    df = pd.DataFrame(rows).drop_duplicates(["day", "title"])
    df["iran"] = df["title"].apply(lambda t: bool(WAR_RE.search(t))); df["conflict"] = df["title"].apply(lambda t: bool(CONFLICT_RE.search(t)))
    df["relevant"] = df["iran"] & df["conflict"]
    sc = df["title"].apply(esc_score); df["esc"] = [s[0] for s in sc]; df["esc_n"] = [s[1] for s in sc]
    return df

def clean_wiki(t):
    t = re.sub(r"<ref[^>]*/>", "", t); t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", t); t = re.sub(r"\[https?://\S+\s?([^\]]*)\]", r"\1", t)
    t = re.sub(r"\{\{[^{}]*\}\}", "", t); t = re.sub(r"'{2,}", "", t); return re.sub(r"<[^>]+>", "", t)

def load_wiki_ce(path="data/wiki_current_events.jsonl"):
    cols = ["day", "ce_items", "ce_iran_items", "ce_iran_chars", "ce_text", "ce_esc"]
    if not os.path.exists(path): return pd.DataFrame(columns=cols)
    rows = []
    for l in open(path):
        j = json.loads(l); t = clean_wiki(j["wikitext"])
        bullets = [re.sub(r"^[*#:]+\s*", "", b).strip() for b in t.split("\n") if re.match(r"^\*{2,}", b)]
        bullets = [b for b in bullets if len(b) > 25]; rel = [b for b in bullets if WAR_RE.search(b)]
        rows.append({"day": j["day"], "ce_items": len(bullets), "ce_iran_items": len(rel), "ce_iran_chars": sum(len(b) for b in rel), "ce_text": " ".join(rel), "ce_esc": np.mean([esc_score(b)[0] for b in rel]) if rel else 0.0})
    return pd.DataFrame(rows, columns=cols)

def finbert_scores(texts, cache="data/finbert_cache.json", batch=64):
    """ProsusAI/finbert tone = P(negative) - P(positive); cached by text."""
    cache_d = json.load(open(cache)) if os.path.exists(cache) else {}
    todo = [t for t in dict.fromkeys(texts) if t not in cache_d]
    if todo:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert"); mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").eval()
        lab = {i: l.lower() for i, l in mdl.config.id2label.items()}
        with torch.no_grad():
            for i in range(0, len(todo), batch):
                b = todo[i:i + batch]; enc = tok(b, padding=True, truncation=True, max_length=64, return_tensors="pt")
                p = torch.softmax(mdl(**enc).logits, -1).numpy()
                for t, pr in zip(b, p):
                    d = {lab[k]: float(pr[k]) for k in range(len(pr))}; cache_d[t] = d["negative"] - d["positive"]
                if (i // batch) % 25 == 0: print("finbert", i, "/", len(todo), flush=True); json.dump(cache_d, open(cache, "w"))
        json.dump(cache_d, open(cache, "w"))
    return [cache_d[t] for t in texts]

STOP = set("the a an and or of to in on for with by at from as is are was were be been it its this that these those after over under into out up down not no than then via amid says said say will would could may might us u.s. new".split())
def _tok(s): return [w for w in re.findall(r"[a-z][a-z'-]{2,}", s.lower()) if w not in STOP]
def novelty(daily_docs, k=5):
    """1 - cosine(TF-IDF of day t headlines, mean TF-IDF of days t-k..t-1): how much new vocabulary the day's war news carries (numpy implementation)."""
    docs = [_tok(d) for d in daily_docs.values]; N = len(docs)
    df = Counter(); [df.update(set(d)) for d in docs]
    vocab = {w: i for i, w in enumerate(w for w, c in df.items() if c >= 2)}
    idf = np.zeros(len(vocab)); 
    for w, i in vocab.items(): idf[i] = math.log((1 + N) / (1 + df[w])) + 1
    X = np.zeros((N, len(vocab)))
    for r, d in enumerate(docs):
        c = Counter(w for w in d if w in vocab)
        for w, n in c.items(): X[r, vocab[w]] = (1 + math.log(n)) * idf[vocab[w]]
        nrm = np.linalg.norm(X[r]); X[r] = X[r] / nrm if nrm > 0 else X[r]
    out = []
    for i in range(N):
        if i == 0 or X[i].sum() == 0: out.append(np.nan); continue
        prev = X[max(0, i - k):i].mean(axis=0); nrm = np.linalg.norm(prev)
        out.append(1 - float(X[i] @ prev / nrm) if nrm > 0 else np.nan)
    return pd.Series(out, index=daily_docs.index)

def build(use_finbert=True):
    g = load_gnews(); ce = load_wiki_ce(); rel = g[g["relevant"]].copy()
    rel["tone"] = finbert_scores(rel["title"].tolist()) if use_finbert else np.nan
    daily = rel.groupby("day").agg(n_rel=("title", "size"), esc_mean=("esc", "mean"), esc_std=("esc", "std"), tone_mean=("tone", "mean"), tone_std=("tone", "std"), text=("title", " || ".join))
    tot = g.groupby("day").agg(n_all=("title", "size"), n_iran=("iran", "sum"))
    daily = tot.join(daily, how="left").join(ce.set_index("day"), how="outer")
    daily["share_rel"] = daily["n_rel"] / daily["n_all"]; daily["novelty"] = novelty(daily["text"].fillna(""))
    tl = pd.read_csv("data/wiki_timeline_daily.csv").set_index("day"); daily["tl_chars"] = tl["nchar"].reindex(daily.index).fillna(0)
    for f in ["timelinevol", "timelinetone"]:
        p = f"data/gdelt_{f}.csv"
        if os.path.exists(p):
            gd = pd.read_csv(p); gd["day"] = pd.to_datetime(gd["date"]).dt.strftime("%Y-%m-%d"); daily["gdelt_" + f[8:]] = gd.groupby("day")["value"].mean()
    daily.index = pd.to_datetime(daily.index); return g, rel, daily

if __name__ == "__main__":
    g, rel, daily = build(use_finbert=("--nofinbert" not in sys.argv))
    daily.to_csv("data/daily_news_features.csv"); rel.to_csv("data/relevant_headlines.csv", index=False); g.to_csv("data/all_headlines.csv", index=False)
    print("headlines:", len(g), "relevant:", len(rel)); print(daily.drop(columns=["text", "ce_text"]).describe().T.round(3))
