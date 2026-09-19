"""Table-1 analogue: for each H day, the most informative headlines, lexicon direction, FinBERT tone, market moves."""
import compat  # noqa: F401  (scipy PROPACK stub, see compat.py)
import sys, re, pandas as pd, numpy as np
tag = sys.argv[1] if len(sys.argv) > 1 else "S1_level_q75"
rel = pd.read_csv("data/relevant_headlines.csv"); rel["day"] = pd.to_datetime(rel["day"])
ch = pd.read_csv("data/changes_2026.csv", index_col=0, parse_dates=True); bdays = ch.index
rel["tday"] = [bdays[bdays.searchsorted(d)] if bdays.searchsorted(d) < len(bdays) else pd.NaT for d in rel["day"]]
tl = pd.read_csv("data/wiki_timeline_daily.csv"); tl["day"] = pd.to_datetime(tl["day"]); tl = tl.set_index("day")["text"]
H = pd.read_csv(f"output/H_days_{tag}.csv"); H["date"] = pd.to_datetime(H["date"]); rows = []
for _, h in H.iterrows():
    d = h["date"]; sub = rel[rel["tday"] == d].copy()
    sub["score"] = sub["esc_n"] + 2 * sub["tone"].abs() + sub["source"].fillna("").str.contains("Reuters|AP News|Bloomberg|CNBC|Financial Times|Wall Street Journal|New York Times|CNN|BBC|Al Jazeera", regex=True).astype(int)
    top = sub.sort_values("score", ascending=False).drop_duplicates("title")["title"].head(3).tolist()
    snippet = " ".join(tl[cd][:300] for cd in pd.date_range(d - pd.Timedelta(days=2 if d.weekday() == 0 else 0), d) if cd in tl.index)
    mk = ch.loc[d, ["UST2Y", "SPX", "BRENT_FUT", "HY_OAS"]]
    rows.append({"date": d.date(), "index": round(h["index"], 2), "risk_direction": h["risk_direction"], "esc_score": round(h["esc_tone"], 2), "finbert_neg": round(h["finbert_tone"], 2) if pd.notna(h["finbert_tone"]) else np.nan,
                 "d2y_bp": round(100 * mk["UST2Y"]), "dSPX_pct": round(mk["SPX"], 2), "dBrent_usd": round(mk["BRENT_FUT"], 2), "dHY_bp": round(100 * mk["HY_OAS"]), "headlines": " | ".join(top), "wiki_timeline": re.sub(r"\s+", " ", snippet)[:400]})
out = pd.DataFrame(rows); out.to_csv(f"output/table1_{tag}.csv", index=False)
pd.set_option("display.max_colwidth", 150); pd.set_option("display.width", 250); print(out[["date", "index", "risk_direction", "d2y_bp", "dSPX_pct", "dBrent_usd", "headlines"]].to_string())
