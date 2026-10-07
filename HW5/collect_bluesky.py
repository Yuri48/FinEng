"""Bluesky posts about the midterms, one calendar day at a time, from the public AppView search endpoint
(app.bsky.feed.searchPosts, English, ranked 'top' so the sample is what was read and shared that day, not the last
hour of the day). The top 100 posts a day for 'midterms' (the search stems, so 'midterm' is included). -> data/bluesky_posts.jsonl (resumable).
The AppView's edge rejects Python's TLS client (403), so requests go through the system curl."""
import json, subprocess, sys, time
from datetime import timedelta
from urllib.parse import urlencode
from config import DATA, START, END, UA

API = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"
OUT = DATA / "bluesky_posts.jsonl"
QUERIES = {"midterms": 1}
PAUSE = 12.0                # the edge firewall returns 403 to bursts; one call every few seconds is tolerated


def page(q, d0, d1, cursor=None):
    p = {"q": q, "limit": 100, "sort": "top", "lang": "en", "since": f"{d0}T00:00:00Z", "until": f"{d1}T00:00:00Z"}
    if cursor: p["cursor"] = cursor
    for attempt in range(5):
        try:
            r = subprocess.run(["curl", "-s", "-m", "60", "-A", UA["User-Agent"], "-w", "\n%{http_code}", f"{API}?{urlencode(p)}"],
                               capture_output=True, text=True, timeout=90)
            body, code = r.stdout.rsplit("\n", 1)
            if code == "200":
                return json.loads(body)
            print("bsky", code, flush=True)
        except Exception as e:
            print("bsky err", e, flush=True)
        time.sleep(90 * (attempt + 1))
    return None


def main(budget_s=None, reverse=False):
    """reverse=True walks back from END into a second file, so two workers can share the sample from both ends."""
    t_start = time.time()
    out = OUT.with_name("bluesky_posts_rev.jsonl") if reverse else OUT
    done = set()
    for f in DATA.glob("bluesky_posts*.jsonl"):
        for line in open(f):
            j = json.loads(line); done.add((j["day"], j["q"]))
    days = [START + timedelta(days=i) for i in range((END - START).days + 1)]
    for d in (reversed(days) if reverse else days):
        if budget_s and time.time() - t_start > budget_s:
            print("time budget reached; rerun to continue", flush=True); return
        for q, npages in QUERIES.items():
            if (d.isoformat(), q) in done:
                continue
            posts, cursor, ok = [], None, True
            for _ in range(npages):
                j = page(q, d, d + timedelta(days=1), cursor)
                if j is None: ok = False; break
                for x in j.get("posts", []):
                    posts.append({"uri": x["uri"], "created_at": x["record"].get("createdAt"), "text": x["record"].get("text", ""),
                                  "likes": x.get("likeCount"), "reposts": x.get("repostCount"), "replies": x.get("replyCount"),
                                  "author": x["author"]["handle"]})
                cursor = j.get("cursor")
                time.sleep(PAUSE)
                if not cursor or not j.get("posts"): break
            if not ok:
                continue
            if any(d.isoformat() in l[:30] for f in DATA.glob("bluesky_posts*.jsonl") for l in open(f)):
                print("met the other worker at", d, flush=True); return
            with open(out, "a") as f:
                f.write(json.dumps({"day": d.isoformat(), "q": q, "posts": posts}) + "\n")
            print("bsky", d, q, len(posts), flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--reverse"]
    main(float(args[0]) if args else None, reverse="--reverse" in sys.argv)      # optional time budget in seconds
