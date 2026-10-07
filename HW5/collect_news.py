"""Google News RSS headlines, one request per query per calendar day (resumable).

    python collect_news.py discovery      # stage 1: two issue-free election queries -> data/news_discovery.jsonl
    python collect_news.py topics [keys]  # stage 2: one query per key topic found by topic_model.py -> data/news_topics*.jsonl
    python collect_news.py equities       # monthly 'midterms + stocks' headlines for basket screening -> data/news_equities.jsonl
"""
import json, os, sys, time, xml.etree.ElementTree as ET
from datetime import timedelta, date
import requests
from config import DATA, START, END, UA, NEWS_DISCOVERY_QUERIES

RSS = "https://news.google.com/rss/search"


def fetch(q, d0, d1):
    p = {"q": f"{q} after:{d0.isoformat()} before:{d1.isoformat()}", "hl": "en-US", "gl": "US", "ceid": "US:en"}
    for attempt in range(5):
        try:
            r = requests.get(RSS, params=p, headers=UA, timeout=60)
            if r.status_code == 200:
                root = ET.fromstring(r.content)
                return [{"title": it.findtext("title"), "pubDate": it.findtext("pubDate"),
                         "source": it.find("source").text if it.find("source") is not None else None,
                         "link": it.findtext("link")} for it in root.findall(".//item")]
            print("status", r.status_code, flush=True); time.sleep(20 * (attempt + 1))
        except Exception as e:
            print("err", e, flush=True); time.sleep(10)
    return None


def run(queries, out, step_days=1, start=START, end=END):
    done = set()
    for f in out.parent.glob(out.name.split(".")[0].split("__")[0] + "*.jsonl"):     # any shard of the same stage
        for line in open(f):
            j = json.loads(line); done.add((j["day"], j["q"]))
    d = start
    while d <= end:
        nd = d + timedelta(days=step_days)
        for qk, q in queries.items():
            if (d.isoformat(), qk) in done:
                continue
            items = fetch(q, d, nd)
            if items is None:
                continue                      # leave the gap; a rerun fills it
            with open(out, "a") as f:
                f.write(json.dumps({"day": d.isoformat(), "q": qk, "items": items}) + "\n")
            print(out.name, d, qk, len(items), flush=True); time.sleep(1.2)
        d = nd


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "discovery"
    if stage == "discovery":
        run(NEWS_DISCOVERY_QUERIES, DATA / "news_discovery.jsonl")
    elif stage == "topics":
        # Written by topic_model.py after the key topics are identified.
        qs = json.load(open(DATA / "topic_queries.json"))
        args = [a for a in sys.argv[2:] if not a.startswith("--")]
        rng = dict(a[2:].split("=") for a in sys.argv[2:] if a.startswith("--"))   # optional --start=YYYY-MM-DD --end=...
        keys = args or list(qs)                               # one process per key (and date range) runs shards in parallel
        s0 = date.fromisoformat(rng["start"]) if "start" in rng else START
        s1 = date.fromisoformat(rng["end"]) if "end" in rng else END
        run({k: qs[k] for k in keys}, DATA / f"news_topics__{'_'.join(keys)}_{s0:%m%d}.jsonl", start=s0, end=s1)
    elif stage == "equities":
        qs = {"stocks": '(midterms OR "midterm elections") (stocks OR sectors OR shares OR investors)',
              "sweep": '("Democratic sweep" OR "blue wave" OR "divided government" OR gridlock) (stocks OR market OR sectors)',
              "winners": '(midterms OR election) ("stocks to buy" OR "winners and losers" OR "stocks that could")'}
        run(qs, DATA / "news_equities.jsonl", step_days=7, start=date(2025, 11, 1))
