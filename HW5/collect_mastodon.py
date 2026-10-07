"""Mastodon posts under the midterm hashtags, paging back through mastodon.social's public tag timelines until the
sample start. -> data/mastodon_posts.jsonl (one line per post; resumable per tag via the oldest id seen)."""
import json, re, time, html
import requests
from config import DATA, START, UA, MASTODON_TAGS

OUT = DATA / "mastodon_posts.jsonl"


def strip(h):
    h = re.sub(r"<br\s*/?>|</p>", " ", h)
    return html.unescape(re.sub(r"<[^>]+>", "", h)).strip()


def main():
    seen, oldest = set(), {}
    if OUT.exists():
        for line in open(OUT):
            j = json.loads(line); seen.add(j["id"])
            if j["tag"] not in oldest or int(j["id"]) < int(oldest[j["tag"]][0]):
                oldest[j["tag"]] = (j["id"], j["created_at"])
    for tag in MASTODON_TAGS:
        max_id = oldest.get(tag, (None, None))[0]
        if oldest.get(tag, (None, "9999"))[1] < START.isoformat():
            continue
        n = 0
        while True:
            p = {"limit": 40}
            if max_id: p["max_id"] = max_id
            try:
                r = requests.get(f"https://mastodon.social/api/v1/timelines/tag/{tag}", params=p, headers=UA, timeout=60)
                if r.status_code == 429:
                    time.sleep(60); continue
                page = r.json()
            except Exception as e:
                print("masto err", e, flush=True); time.sleep(10); continue
            if not page:
                break
            with open(OUT, "a") as f:
                for s in page:
                    if s["id"] in seen: continue
                    seen.add(s["id"])
                    f.write(json.dumps({"id": s["id"], "tag": tag, "created_at": s["created_at"], "lang": s.get("language"),
                                        "text": strip(s.get("content") or ""), "reblogs": s.get("reblogs_count"),
                                        "favs": s.get("favourites_count"), "acct": s["account"]["acct"]}) + "\n")
                    n += 1
            max_id = page[-1]["id"]
            print("masto", tag, page[-1]["created_at"][:10], n, flush=True)
            if page[-1]["created_at"][:10] < START.isoformat():
                break
            time.sleep(1.1)


if __name__ == "__main__":
    main()
