"""Load every collected source into one document table.

Columns: doc_id, source (news | bluesky | mastodon | reddit | lemmy), kind (news | social), date (US/Eastern calendar day),
text (cleaned, what the models see), weight (engagement, used only for engagement-weighted robustness), lean (Reddit
community lean), community.
"""
import json, re, html
from email.utils import parsedate_to_datetime
from pathlib import Path
import numpy as np
import pandas as pd
from config import DATA, START, END, SUB_LEAN

ET = "America/New_York"
URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"(?<!\w)@[\w.\-]+")


def _clean(t):
    t = html.unescape(t or "")
    t = "\n".join(l for l in t.split("\n") if not l.lstrip().startswith(">"))      # drop quoted parent text
    t = URL.sub(" ", t); t = MENTION.sub(" ", t)
    t = re.sub(r"[#*_~`|\[\]]+", " ", t)                                           # hashtag signs, markdown
    t = re.sub(r"\(\s*\)", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _et_date(ts, unit=None):
    s = pd.to_datetime(ts, unit=unit, utc=True, errors="coerce")
    return s.dt.tz_convert(ET).dt.date


def _pub_ok(pub, day, window=1):
    """Guard against queries for which Google ignores the after:/before: operators and returns today's results:
    keep an item only if it was published inside the window it was requested for (plus one day of slack)."""
    try:
        p = parsedate_to_datetime(pub).astimezone().date()
    except Exception:
        return False
    return -1 <= (p - pd.Timestamp(day).date()).days <= window


def load_news(path=DATA / "news_discovery.jsonl", tag="news", window=1):
    rows = []
    for line in open(path):
        j = json.loads(line)
        for it in j["items"]:
            if not _pub_ok(it.get("pubDate"), j["day"], window):
                continue
            title = re.sub(r"\s+-\s+[^-]+$", "", it.get("title") or "").strip()      # drop ' - Publisher'
            rows.append({"date": j["day"], "text": title, "community": it.get("source") or "", "q": j["q"]})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df.text.str.len() >= 20].sort_values("date").drop_duplicates(["q", "text"] if tag == "news2" else "text")
    df["source"], df["kind"], df["weight"], df["lean"] = tag, "news", 1.0, ""
    df["doc_id"] = [f"{tag}_{i}" for i in range(len(df))]
    return df


def load_bluesky(pattern="bluesky_posts*.jsonl"):
    """Both collector files: the forward worker (bluesky_posts.jsonl) and the reverse worker (bluesky_posts_rev.jsonl)."""
    files = sorted(DATA.glob(pattern))
    if not files: return pd.DataFrame()
    rows = []
    for line in (l for f in files for l in open(f)):
        j = json.loads(line)
        for p in j["posts"]:
            rows.append({"doc_id": "bsky_" + p["uri"].split("/")[-1], "created": p["created_at"], "raw": p["text"],
                         "weight": 1 + (p.get("likes") or 0) + 2 * (p.get("reposts") or 0), "community": ""})
    df = pd.DataFrame(rows).drop_duplicates("doc_id")
    df["date"] = _et_date(df.created)
    df["source"], df["kind"], df["lean"] = "bluesky", "social", ""
    return df


def load_mastodon(path=DATA / "mastodon_posts.jsonl"):
    df = pd.DataFrame([json.loads(l) for l in open(path)]).drop_duplicates("id")
    df = df[df.lang.isin(["en", None]) | df.lang.isna()]
    df["doc_id"] = "masto_" + df.id.astype(str)
    df["date"] = _et_date(df.created_at)
    df["raw"], df["weight"], df["community"] = df.text, 1 + df.favs.fillna(0) + 2 * df.reblogs.fillna(0), ""
    df["source"], df["kind"], df["lean"] = "mastodon", "social", ""
    return df


def load_reddit(folder=DATA / "reddit"):
    rows = []
    for f in sorted(Path(folder).glob("*.jsonl")):
        for line in open(f):
            j = json.loads(line)
            for c in j["items"]:
                rows.append({"doc_id": "rd_" + c["id"], "created": c["created_utc"], "raw": c["body"],
                             "weight": max(1, (c.get("score") or 1)), "community": j["sub"], "lean": SUB_LEAN.get(j["sub"], "")})
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("doc_id")
    df = df[~df.raw.isin(["[deleted]", "[removed]"])]
    df["date"] = _et_date(df.created, unit="s")
    df["source"], df["kind"] = "reddit", "social"
    return df


def load_lemmy(path=DATA / "lemmy.jsonl"):
    if not Path(path).exists(): return pd.DataFrame()
    df = pd.DataFrame([json.loads(l) for l in open(path)]).drop_duplicates("id")
    df["doc_id"] = ["lemmy_" + str(i) for i in range(len(df))]
    df["date"] = _et_date(df.created_at)
    df["raw"], df["weight"] = df.text, 1 + df.score.clip(lower=0)
    df["source"], df["kind"], df["lean"] = "lemmy", "social", ""
    return df


def load_corpus(min_chars=25, max_chars=1500):
    social = [d for d in (load_bluesky(), load_mastodon(), load_reddit(), load_lemmy()) if len(d)]
    soc = pd.concat(social, ignore_index=True)
    soc["text"] = soc.raw.map(_clean).str.slice(0, max_chars)
    soc = soc[soc.text.str.len() >= min_chars]
    soc = soc[soc.text.str.contains(r"midterm", case=False)]          # every social document must mention the midterms
    soc = soc.drop_duplicates("text")                                  # cross-posts and bot reposts
    news = load_news()
    cols = ["doc_id", "source", "kind", "date", "text", "weight", "lean", "community"]
    df = pd.concat([news[cols], soc[cols]], ignore_index=True)
    df = df[(df.date >= START) & (df.date <= END)].reset_index(drop=True)
    df["week"] = pd.to_datetime(df.date).dt.to_period("W-SUN").dt.end_time.dt.normalize().dt.date   # week ending Sunday
    return df


if __name__ == "__main__":
    df = load_corpus()
    print(df.groupby(["source"]).agg(n=("doc_id", "size"), first=("date", "min"), last=("date", "max"),
                                     med_chars=("text", lambda s: int(s.str.len().median()))))
    print(len(df))
