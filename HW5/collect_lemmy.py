"""Lemmy (lemmy.world, federated view) comments and posts that mention the midterms, newest first, paging back to the
sample start. -> data/lemmy.jsonl (one line per item)."""
import json, time
import requests
from config import DATA, START, UA

API = "https://lemmy.world/api/v3/search"
OUT = DATA / "lemmy.jsonl"


def main():
    seen = set()
    with open(OUT, "w") as f:
        for kind in ("Comments", "Posts"):
            page = 1
            while True:
                try:
                    r = requests.get(API, params={"q": "midterm", "type_": kind, "sort": "New", "limit": 50, "page": page},
                                     headers=UA, timeout=60)
                    j = r.json()
                except Exception as e:
                    print("lemmy err", e, flush=True); time.sleep(10); continue
                items = j.get("comments" if kind == "Comments" else "posts", [])
                if not items:
                    break
                for it in items:
                    if kind == "Comments":
                        c = it["comment"]; rec = {"id": c["ap_id"], "kind": "comment", "created_at": c["published"],
                                                  "text": c["content"], "score": it["counts"]["score"],
                                                  "community": it["community"]["name"]}
                    else:
                        p = it["post"]; rec = {"id": p["ap_id"], "kind": "post", "created_at": p["published"],
                                               "text": (p["name"] + ". " + (p.get("body") or "")).strip(),
                                               "score": it["counts"]["score"], "community": it["community"]["name"]}
                    if rec["id"] in seen: continue
                    seen.add(rec["id"]); f.write(json.dumps(rec) + "\n")
                oldest = items[-1]["comment" if kind == "Comments" else "post"]["published"][:10]
                print("lemmy", kind, page, oldest, len(seen), flush=True)
                if oldest < START.isoformat():
                    break
                page += 1; time.sleep(1.0)


if __name__ == "__main__":
    main()
