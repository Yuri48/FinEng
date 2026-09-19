"""
Driver: composite war-news index -> H/L day selection -> Rigobon-Sack estimates for all variables -> tables & figures.
Specifications:
  S1 baseline : full sample (2 Jan - 16 Sep 2026), H = top 25% of the NLP index, L = nearest non-H business day (paper's rule)
  S2 tighter  : H = top 15%
  S3 surprise : H = top 25% of (index - trailing 10-business-day median): intensity relative to the prevailing regime
  S4 pre-war  : 2 Jan - 27 Feb only, H = top 36% (17/47 as in the paper): closest analogue to the 2003 'risk of war' setting
  S5 war/post : 2 Mar - 16 Sep, H = top 25%
  S6 curated  : hand-collected event dates (the paper's 'reading newspapers' approach) for comparison
  Each also reported with Brent (+$5) as the normalising variable.
"""
import compat  # noqa: F401  (scipy PROPACK stub, see compat.py)
import json, sys, os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from warrisk import rs_estimate, pick_L_days, table3_all_days
OUT = "output"; os.makedirs(OUT + "/fig", exist_ok=True)
ch = pd.read_csv("data/changes_2026.csv", index_col=0, parse_dates=True); spec = json.load(open("data/var_spec.json"))
feat = pd.read_csv("data/daily_news_features.csv", index_col=0, parse_dates=True)
PAPER = ["UST10Y", "BEI10Y", "SPX", "BBB_OAS", "HY_OAS", "WTI_FUT", "GOLD", "USD_BROAD"]
ORDER = PAPER + ["TIPS10Y", "BRENT_FUT", "DXY", "EURUSD", "USDJPY", "USDCHF", "VIX", "TLT", "HYG", "STOXX50", "NIKKEI", "EEM", "TA35", "KSA", "ITA", "JETS", "XLE", "NATGAS", "BDRY", "BTC"]

def z(s): return (s - s.mean()) / s.std()
comp = pd.DataFrame(index=feat.index)
comp["volume"] = z(np.log1p(feat["n_rel"].fillna(0)))                       # relevant headline count (RSS caps ~100/query)
comp["share"] = z(feat["share_rel"].fillna(0))                              # share of Iran coverage that is conflict-related
comp["wiki_ce"] = z(np.log1p(feat["ce_iran_chars"].fillna(0)))              # Wikipedia current-events Iran text volume
comp["wiki_tl"] = z(np.log1p(feat["tl_chars"].fillna(0)))                   # Wikipedia war-timeline daily text volume
comp["novelty"] = z(feat["novelty"].fillna(feat["novelty"].median()))      # new vocabulary vs prior 5 days
comp["esc_abs_chg"] = z(feat["esc_mean"].fillna(0).diff().abs().fillna(0)) # swing in escalation direction
comp["tone_disp"] = z(feat["tone_std"].fillna(feat["tone_std"].median()))  # dispersion of FinBERT tone (disagreement)
if "gdelt_vol" in feat: comp["gdelt_vol"] = z(np.log1p(feat["gdelt_vol"].fillna(feat["gdelt_vol"].median())))
COMPONENTS = list(comp.columns); comp["index"] = comp[COMPONENTS].mean(axis=1)
comp["volume_idx"] = comp[["volume", "wiki_ce", "wiki_tl"]].mean(axis=1)
comp["esc_mean"] = feat["esc_mean"]; comp["tone_mean"] = feat["tone_mean"]
bdays = ch.index
def next_bday(d):
    i = bdays.searchsorted(d); return bdays[i] if i < len(bdays) else pd.NaT
comp["tday"] = [next_bday(d) for d in comp.index]
td = comp.dropna(subset=["tday"]).groupby("tday").agg(index=("index", "max"), volume_idx=("volume_idx", "max"), esc_mean=("esc_mean", "mean"), tone_mean=("tone_mean", "mean"), ndays=("index", "size"))
td = td.reindex(bdays).dropna(subset=["index"]); td["index"] = z(td["index"]); td["volume_idx"] = z(td["volume_idx"])
td["surprise"] = (td["index"] - td["index"].rolling(10, min_periods=3).median().shift(1)).fillna(td["index"])
td.to_csv(f"{OUT}/index_trading_days.csv"); comp.to_csv(f"{OUT}/index_calendar_days.csv")

def select(col, q, start=None, end=None):
    s = td[col].loc[start:end]; thr = s.quantile(q); return list(s.index[s >= thr]), thr
SPECS = {"S1_level_q75": dict(col="index", q=0.75), "S2_level_q85": dict(col="index", q=0.85), "S3_surprise_q75": dict(col="surprise", q=0.75),
         "S4_prewar": dict(col="index", q=1 - 17 / 47, start="2026-01-02", end="2026-02-27"), "S5_warpost": dict(col="index", q=0.75, start="2026-03-02", end="2026-09-16"), "S6_curated": dict(col="index", q=None), "S7_volume_q75": dict(col="volume_idx", q=0.75)}
cur = pd.read_csv("data/curated_events.csv"); cur["date"] = pd.to_datetime(cur["date"])
def run(H, L, base, scale, window):
    rows = []; bd = [d for d in bdays if window[0] <= d <= window[1]]; nH_all = len(H); nL_all = len(bd) - nH_all; chw = ch.loc[window[0]:window[1]]
    for v in ORDER:
        if v == base: continue
        r = rs_estimate(chw, v, base, H, L, scale=scale); d = r["nu3"] / scale
        rows.append({"variable": spec[v]["label"], "code": v, "units": spec[v]["units"], "IV nu1": r["nu1"], "t1": r["nu1_t"], "IV nu2": r["nu2"], "t2": r["nu2_t"], "IV nu3": r["nu3"], "t3": r["nu3_t"],
                     "Var L": r["varL"], "Var H": r["varH"], "Pred dVar": r["pred_dvar"], "% expl H": r["pct_H"], "% expl all": (table3_all_days(chw, v, base, H, L, d, nH_all, nL_all) if pd.notna(r["pct_H"]) else np.nan), "dVar(base)": r["dVar_x1"], "dCov": r["dCov"], "nH": r["nH"], "nL": r["nL"]})
    return pd.DataFrame(rows)
results = {}
for tag, sp in SPECS.items():
    window = (pd.Timestamp(sp.get("start") or bdays[0]), pd.Timestamp(sp.get("end") or bdays[-1])); bd = [d for d in bdays if window[0] <= d <= window[1]]
    if tag == "S6_curated": H = [d for d in cur["date"] if d in td.index]; thr = np.nan
    else: H, thr = select(sp["col"], sp["q"], sp.get("start"), sp.get("end"))
    L = pick_L_days(bd, H)
    sel = pd.DataFrame({"date": H, "index": td.loc[H, "index"].values, "surprise": td.loc[H, "surprise"].values, "esc_tone": td.loc[H, "esc_mean"].values, "finbert_tone": td.loc[H, "tone_mean"].values})
    sel["risk_direction"] = np.where(sel["esc_tone"] > 0.2, "Increased", np.where(sel["esc_tone"] < -0.2, "Decreased", "Unclear"))
    sel.to_csv(f"{OUT}/H_days_{tag}.csv", index=False); pd.Series(L, name="date").to_csv(f"{OUT}/L_days_{tag}.csv", index=False)
    v1L = float(np.mean(ch.loc[L, "UST2Y"] ** 2)); v1H = float(np.mean(ch.loc[H, "UST2Y"] ** 2)); bL = float(np.nanmean(ch.loc[L, "BRENT_FUT"] ** 2)); bH = float(np.nanmean(ch.loc[H, "BRENT_FUT"] ** 2))
    print(f"\n===== {tag}: {window[0].date()}..{window[1].date()}  nH={len(H)} nL={len(L)} of {len(bd)} days | 2y var L {v1L:.5f} H {v1H:.5f} | Brent var L {bL:.2f} H {bH:.2f}")
    t_ust = run(H, L, "UST2Y", -0.25, window); t_brent = run(H, L, "BRENT_FUT", 5.0, window)
    t_ust.to_csv(f"{OUT}/table2_{tag}_ust2y.csv", index=False); t_brent.to_csv(f"{OUT}/table2_{tag}_brent.csv", index=False)
    results[tag] = dict(H=H, L=L, ust=t_ust, brent=t_brent, window=window, v1=(v1L, v1H), vb=(bL, bH))
    print(t_ust[["variable", "units", "IV nu1", "t1", "IV nu2", "t2", "IV nu3", "t3", "% expl H", "% expl all"]].round(3).to_string())
    print("-- Brent-normalised (nu3, t):"); print(t_brent[["variable", "units", "IV nu3", "t3", "% expl H", "% expl all"]].round(3).to_string())
for base in ["ust", "brent"]:
    summ = []
    for v in (PAPER if base == "ust" else [c for c in ORDER if c != "BRENT_FUT"]):
        row = {"variable": spec[v]["label"], "units": spec[v]["units"]}
        for tag in SPECS:
            t = results[tag][base].set_index("code"); row[tag] = f'{t.loc[v, "IV nu3"]:.3g} ({t.loc[v, "t3"]:.2f})'
        summ.append(row)
    pd.DataFrame(summ).to_csv(f"{OUT}/summary_across_specs_{base}.csv", index=False); print(f"\n== summary ({base}) ==\n", pd.DataFrame(summ).to_string())
json.dump({k: {"H": [str(d.date()) for d in v["H"]], "L": [str(d.date()) for d in v["L"]], "window": [str(v["window"][0].date()), str(v["window"][1].date())], "var2y_L_H": v["v1"], "varBrent_L_H": v["vb"]} for k, v in results.items()}, open(f"{OUT}/specs.json", "w"), indent=1)
# ---- figures
lv = pd.read_csv("data/levels.csv", index_col=0, parse_dates=True).loc["2026-01-01":]
EV = [("2026-02-28", "strikes begin"), ("2026-04-08", "ceasefire"), ("2026-06-17", "MoU signed"), ("2026-07-08", "ceasefire over"), ("2026-08-17", "MoU expires")]
for tag in ["S1_level_q75", "S3_surprise_q75", "S6_curated", "S7_volume_q75"]:
    H = results[tag]["H"]; col = "surprise" if "surprise" in tag else ("volume_idx" if "volume" in tag else "index")
    fig, ax = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
    ax[0].plot(td.index, td[col], color="k", lw=1); ax[0].scatter(H, td.loc[H, col], color="crimson", s=20, zorder=3, label=f"H days ({tag}, n={len(H)})"); ax[0].set_ylabel("NLP war-news index"); ax[0].legend(loc="upper right")
    for a, c, lab in [(ax[1], "BZ=F", "Brent front future ($/bbl)"), (ax[2], "UST2Y", "2y Treasury yield (%)"), (ax[3], "^GSPC", "S&P 500"), (ax[4], "HY_OAS", "High-yield OAS (%)")]:
        sr = lv[c].dropna(); a.plot(sr.index, sr.values, lw=1); a.set_ylabel(lab)
        for h in H: a.axvspan(h - pd.Timedelta(hours=12), h + pd.Timedelta(hours=12), color="crimson", alpha=.12)
    for a in ax:
        for d, lab in EV: a.axvline(pd.Timestamp(d), color="navy", lw=.7, ls=":")
    for d, lab in EV: ax[0].text(pd.Timestamp(d), ax[0].get_ylim()[1], lab, rotation=90, va="top", ha="right", fontsize=7, color="navy")
    fig.suptitle("Iran war news intensity (NLP index) and markets, 2026 - shaded = high-variance (H) days"); fig.tight_layout(); fig.savefig(f"{OUT}/fig/index_and_markets_{tag}.png", dpi=130); plt.close(fig)
fig, a = plt.subplots(figsize=(13, 4)); comp[COMPONENTS].rolling(3).mean().plot(ax=a, lw=.8); a.set_title("Components of the daily war-news index (3-day MA, z-scores)"); fig.tight_layout(); fig.savefig(f"{OUT}/fig/index_components.png", dpi=130); plt.close(fig)
fig, a = plt.subplots(figsize=(13, 3.5)); a.plot(feat.index, feat["esc_mean"].rolling(3).mean(), label="escalation lexicon score (3d MA)"); a.plot(feat.index, feat["tone_mean"].rolling(3).mean(), label="FinBERT negativity (3d MA)"); a.axhline(0, c="k", lw=.5); a.legend(); a.set_title("Direction of Iran-war news"); fig.tight_layout(); fig.savefig(f"{OUT}/fig/tone.png", dpi=130); plt.close(fig)
for tag in SPECS:
    for base, ttl in [("ust", "war-risk rise that lowers the 2y yield 25bp"), ("brent", "war-risk rise that raises Brent $5")]:
        t = results[tag][base].iloc[::-1]; fig, a = plt.subplots(figsize=(9, 9))
        a.barh(t["variable"] + " (" + t["units"] + ")", t["IV nu3"], color=np.where(t["t3"] > 1.96, "crimson", "lightgray")); a.axvline(0, c="k", lw=.8); a.set_xscale("symlog", linthresh=1)
        a.set_title(f"{tag}: response to a {ttl}\n(red: |t|>1.96, combined-instrument IV)"); fig.tight_layout(); fig.savefig(f"{OUT}/fig/coefficients_{tag}_{base}.png", dpi=130); plt.close(fig)
print("done")
