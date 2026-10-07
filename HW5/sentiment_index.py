"""Step 2b: issue salience and the weekly Key-Topic Sentiment Index (KTSI).

Issue of a document: the issue its BERTopic cluster was mapped to (data/topic_labels.json) for the discovery corpus,
or the issue of the query for the stage-2 issue-anchored headlines.

Salience: share of each issue among documents that discuss an issue (campaign/horse-race and noise excluded), by
source; the 'voter' share averages the four social platforms equally so that no single platform dominates.

Weekly sentiment of issue k (two measures, tone_rob and pro_dem):
    x_d        = score_d - mean(score | source, issue)        remove each platform's baseline for that issue
    S_{k,w,s}  = mean of x_d over documents of issue k, week w, source s   (cells with >= MIN_N documents)
    S_{k,w}    = mean over sources of S_{k,w,s}                each source counts equally every week
KTSI_w = sum_k s_k S_{k,w} / sum_k s_k (over issues observed that week), s_k = voter salience of key issue k.
Daily versions use trailing 7-day windows. -> data/issue_week.csv, data/ktsi_weekly.csv, data/ktsi_daily.csv,
output/table_salience.csv
"""
import json
import numpy as np
import pandas as pd
from config import DATA, OUT

MIN_N = 5
SOCIAL = ["bluesky", "mastodon", "reddit", "lemmy"]
NON_ISSUES = {"Campaign & horse race", "Other / noise"}
KEY_ISSUES = ["Economy & cost of living", "Democracy & election integrity", "Iran war & foreign policy",
              "Corruption & accountability", "Immigration & ICE", "Health care", "AI, tech & data centres", "Tariffs & trade"]
POLL_BGSU = {"Economy & cost of living": "economy 38%, inflation 31%", "Corruption & accountability": "government corruption 30%",
             "Immigration & ICE": "immigration 25%", "Democracy & election integrity": "threats to democracy 24%",
             "Health care": "health care 22%", "Iran war & foreign policy": "Middle East conflict 10%, foreign affairs 6%",
             "Tariffs & trade": "(not listed; taxes/fiscal 12%)", "AI, tech & data centres": "AI 8%, data centres 5%",
             "Climate & energy": "climate 11%, energy 2%", "Abortion & social issues": "abortion 6%, LGBTQ+ 3%",
             "Crime & political violence": "crime 10%", "Redistricting & voting rights": "(not listed)"}


def load():
    s = pd.read_parquet(DATA / "doc_sentiment.parquet")
    lab = json.load(open(DATA / "topic_labels.json"))
    s["issue"] = s.topic.map(lambda t: lab.get(str(int(t)), {}).get("issue") if pd.notna(t) else None)
    s.loc[s.issue_q.notna(), "issue"] = s.loc[s.issue_q.notna(), "issue_q"]
    s.loc[s.topic == -1, "issue"] = s.loc[s.topic == -1, "issue"].fillna("Other / noise")
    s["source2"] = s.source.replace({"news2": "news"})
    s["date"] = pd.to_datetime(s.date)
    s["week"] = pd.to_datetime(s.week)
    return s


def salience(s):
    d = s[(s.source != "news2") & s.issue.notna() & ~s.issue.isin(NON_ISSUES)]       # discovery corpus only
    tab = d.groupby(["issue", "source"]).size().unstack(fill_value=0)
    share = tab / tab.sum()
    out = pd.DataFrame({"n_docs": tab.sum(axis=1)})
    for c in share.columns: out[f"share_{c}"] = share[c]
    out["voter_share"] = share[[c for c in SOCIAL if c in share]].mean(axis=1)        # platforms weighted equally
    out["news_share"] = share.get("news", np.nan)
    allshare = s[(s.source != "news2") & s.issue.notna()].groupby("issue").size()
    out["share_of_all_docs"] = allshare / allshare.sum()
    out["poll_BGSU_Sep2026"] = out.index.map(POLL_BGSU)
    out = out.sort_values("voter_share", ascending=False)
    out["rank_voter"] = np.arange(1, len(out) + 1)
    out["rank_news"] = out.news_share.rank(ascending=False).astype(int)
    out.to_csv(OUT / "table_salience.csv")
    return out


def weekly(s, col, freq="week"):
    d = s[s.issue.isin(KEY_ISSUES) & s[col].notna()].copy()
    d["x"] = d[col] - d.groupby(["source2", "issue"])[col].transform("mean")
    cell = d.groupby(["issue", freq, "source2"]).x.agg(["mean", "size"]).reset_index()
    cell = cell[cell["size"] >= MIN_N]
    return cell.groupby(["issue", freq])["mean"].mean().unstack(0)                   # sources equally weighted


def composite(S, w):
    W = pd.DataFrame(np.tile(w.reindex(S.columns).values, (len(S), 1)), index=S.index, columns=S.columns).where(S.notna())
    return (S * W).sum(axis=1) / W.sum(axis=1)


def daily(s, col, window=7):
    """Trailing 7-day version: each day pools the documents of the past 7 days."""
    d = s[s.issue.isin(KEY_ISSUES) & s[col].notna()].copy()
    d["x"] = d[col] - d.groupby(["source2", "issue"])[col].transform("mean")
    days = pd.date_range(d.date.min() + pd.Timedelta(days=window - 1), d.date.max())
    out = {}
    g = {k: v for k, v in d.groupby("issue")}
    for k, v in g.items():
        cell = v.groupby(["date", "source2"]).x.agg(["sum", "size"]).unstack(1).fillna(0)
        cell = cell.reindex(pd.date_range(d.date.min(), d.date.max()), fill_value=0)
        roll = cell.rolling(window).sum()
        m = roll["sum"] / roll["size"].where(roll["size"] >= MIN_N)
        out[k] = m.mean(axis=1)
    return pd.DataFrame(out).loc[days]


if __name__ == "__main__":
    s = load()
    sal = salience(s)
    pd.set_option("display.width", 250)
    print(sal.round(3).drop(columns=["poll_BGSU_Sep2026"]).to_string())
    w = sal.voter_share.reindex(KEY_ISSUES)
    w = w / w.sum()
    print("KTSI weights:", w.round(3).to_dict())
    tone_w, pro_w = weekly(s, "tone_rob"), weekly(s, "pro_dem")
    vader_w = weekly(s, "tone_vader")
    # issue x week counts, for the appendix
    cnt = s[s.issue.isin(KEY_ISSUES)].groupby(["week", "issue"]).size().unstack(1)
    iw = pd.concat({"tone": tone_w, "pro_dem": pro_w, "vader": vader_w, "n": cnt}, axis=1)
    iw.to_csv(DATA / "issue_week.csv")
    k = pd.DataFrame({"ktsi_tone": composite(tone_w, w), "ktsi_prodem": composite(pro_w, w), "ktsi_vader": composite(vader_w, w),
                      "ktsi_tone_eqw": tone_w.mean(axis=1), "ktsi_prodem_eqw": pro_w.mean(axis=1)})
    k.index.name = "week"
    k.to_csv(DATA / "ktsi_weekly.csv")
    kd = pd.DataFrame({"ktsi_tone": composite(daily(s, "tone_rob"), w), "ktsi_prodem": composite(daily(s, "pro_dem"), w)})
    kd.index.name = "date"
    kd.to_csv(DATA / "ktsi_daily.csv")
    json.dump(w.to_dict(), open(DATA / "ktsi_weights.json", "w"), indent=1)
    print(k.round(3).tail(10).to_string())
