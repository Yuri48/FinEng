"""Reddit comments that mention the midterms, from the Arctic Shift archive of the Reddit API (resumable).
One call per subreddit per window (config.SUBREDDITS gives the window design), newest first, 100 comments per call.
Windows are visited in a fixed random order, so a run cut short by the rate limit still covers the whole sample.
-> data/reddit/<subreddit>.jsonl  (one line per call: window, subreddit, list of comments)
    python collect_reddit.py [sub1 sub2 ...]     # default: all subreddits in config.SUBREDDITS"""
import json, os, random, sys, time
from datetime import datetime, timedelta, timezone
import requests
from config import DATA, START, END, UA, SUBREDDITS

API = "https://arctic-shift.photon-reddit.com/api/comments/search"
(DATA / "reddit").mkdir(exist_ok=True)
KEEP = ("id", "created_utc", "score", "body", "author", "link_id", "parent_id")


def fetch(sub, t0, t1):
    p = {"subreddit": sub, "body": "midterm", "after": t0.isoformat(), "before": t1.isoformat(), "limit": 100, "sort": "desc"}
    for attempt in range(6):
        try:
            r = requests.get(API, params=p, headers=UA, timeout=90)
            j = r.json()
            if r.status_code == 200 and j.get("data") is not None:
                return [{k: c.get(k) for k in KEEP} for c in j["data"]]
            wait = int(r.headers.get("x-ratelimit-reset", 5)) + 1 if r.status_code == 429 else 10 * (attempt + 1)
            print("arctic", r.status_code, j.get("error"), "wait", wait, flush=True)
        except Exception as e:
            print("arctic err", e, flush=True); wait = 10
        time.sleep(wait)
    return None


def main(subs):
    done = set()
    for sub in subs:
        out = DATA / "reddit" / f"{sub}.jsonl"
        if out.exists():
            for line in open(out):
                j = json.loads(line); done.add((j["t0"], j["sub"]))
    start = datetime(START.year, START.month, START.day, tzinfo=timezone.utc)
    end = datetime(END.year, END.month, END.day, tzinfo=timezone.utc) + timedelta(days=1)
    windows = []                                   # (t0, t1, sub)
    for sub in subs:
        length, h0, step = SUBREDDITS[sub]
        t0 = start + timedelta(hours=h0)
        while t0 < end:
            windows.append((t0, min(t0 + timedelta(hours=length), end), sub)); t0 += timedelta(hours=step)
    random.Random(7).shuffle(windows)
    for t0, t1, sub in windows:
        key = t0.strftime("%Y-%m-%dT%H:%M")
        if (key, sub) in done:
            continue
        items = fetch(sub, t0.replace(tzinfo=None), t1.replace(tzinfo=None))
        if items is None:
            continue
        with open(DATA / "reddit" / f"{sub}.jsonl", "a") as f:
            f.write(json.dumps({"t0": key, "t1": t1.strftime("%Y-%m-%dT%H:%M"), "sub": sub, "items": items}) + "\n")
        print("reddit", key, sub, len(items), flush=True); time.sleep(4.0)


if __name__ == "__main__":
    main(sys.argv[1:] or list(SUBREDDITS))
