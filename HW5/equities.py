"""Step 3a: the 'midterm basket' and its market-adjusted performance.

Market model.  For each stock, r_it = a_i + b_i r_SPY,t + e_it on daily simple returns over 2025 (the estimation
               window; CRCL from its June 2025 IPO). Abnormal return in 2026: AR_it = r_it - a_i - b_i r_SPY,t.
Basket.        Equal weight within each policy channel, equal weight across channels within each leg;
               basket AR = (Democratic-sweep winners leg) - (Republican-hold winners leg). Cumulative AR from 2 Jan 2026.
Election beta. AR_it regressed on the change in P(Democratic sweep) between 16:00 ET prints (mean of Polymarket and
               Kalshi), daily and weekly (Friday to Friday), Newey-West standard errors. A beta of 0.10 means a 10-point
               rise in the sweep probability comes with a 1% abnormal return.
-> output/table_equities.csv, output/table_channels.csv, data/basket_daily.csv
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from config import DATA, OUT

EST = ("2025-01-02", "2025-12-31")
EVT = ("2026-01-02", "2026-10-06")


def load():
    px = pd.read_csv(DATA / "prices.csv", index_col=0, parse_dates=True)
    r = px.pct_change(fill_method=None)
    uni = pd.read_csv(DATA / "universe.csv")
    pm = pd.read_csv(DATA / "pm_4pm.csv", index_col=0, parse_dates=True)
    pm["p"] = pm[["poly_DD_4pm", "kalshi_DD_4pm"]].mean(axis=1)
    pm = pm[pm.index.isin(px.index)]          # trading days only, so a holiday's move counts on the next session
    return px, r, uni, pm


def market_model(r, uni):
    rows, ar = [], {}
    for t in uni.ticker:
        d = pd.concat([r[t], r["SPY"]], axis=1, keys=["y", "m"]).loc[EST[0]:EST[1]].dropna()
        f = sm.OLS(d.y, sm.add_constant(d.m)).fit()
        a, b = f.params["const"], f.params["m"]
        ar[t] = (r[t] - a - b * r["SPY"]).loc[EVT[0]:EVT[1]]
        rows.append({"ticker": t, "alpha_2025": a, "beta_2025": b, "n_est": len(d), "resid_vol_2025": f.resid.std() * np.sqrt(252)})
    return pd.DataFrame(rows).set_index("ticker"), pd.DataFrame(ar)


def basket(ar, uni):
    ch_of = uni.set_index("ticker").channel
    ch = ar.T.groupby(ch_of).mean().T                                   # channel AR, equal weight within channel
    sign = uni.groupby("channel").sign.first()
    long = ch[sign[sign > 0].index].mean(axis=1); short = ch[sign[sign < 0].index].mean(axis=1)
    return pd.DataFrame({"long": long, "short": short, "basket": long - short}), ch


def hac(y, x, lags):
    d = pd.concat([y, x], axis=1).dropna()
    f = sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1:])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return f


def election_betas(ar, ch, bsk, pm):
    dp = pm["p"].diff()                                                 # 4pm-to-4pm change, trading days
    dp_w = pm["p"].resample("W-FRI").last().diff()
    rows = []
    series = {**{t: ar[t] for t in ar}, **{f"[{c}]": ch[c] for c in ch}, "[BASKET]": bsk["basket"],
              "[LONG leg]": bsk["long"], "[SHORT leg]": bsk["short"]}
    for name, s in series.items():
        fd = hac(s, dp.rename("dp"), 5)
        sw = s.resample("W-FRI").sum(min_count=3)
        fw = hac(sw, dp_w.rename("dp"), 2)
        rows.append({"name": name, "beta_daily": fd.params["dp"], "t_daily": fd.tvalues["dp"], "n_daily": int(fd.nobs),
                     "beta_weekly": fw.params["dp"], "t_weekly": fw.tvalues["dp"], "n_weekly": int(fw.nobs),
                     "car_2026_pct": 100 * s.sum()})
    return pd.DataFrame(rows).set_index("name")


def stability(ar, pm, split="2026-06-01"):
    """Is there a stable midterm factor? Weekly election betas estimated Jan-May vs Jun-Oct, rank correlation."""
    dp_w = pm["p"].resample("W-FRI").last().diff().rename("dp")
    b = {}
    for t in ar:
        sw = ar[t].resample("W-FRI").sum(min_count=3)
        b[t] = [hac(sw.loc[:split], dp_w.loc[:split], 2).params["dp"], hac(sw.loc[split:], dp_w.loc[split:], 2).params["dp"]]
    b = pd.DataFrame(b, index=["beta_H1", "beta_H2"]).T
    rho, p = __import__("scipy").stats.spearmanr(b.beta_H1, b.beta_H2)
    return b, rho, p


def event_days(bsk, pm, k=12):
    """Basket abnormal return on the k trading days with the largest 4pm-to-4pm moves in P(sweep)."""
    dp = pm["p"].diff().loc[EVT[0]:EVT[1]]
    top = dp.abs().sort_values(ascending=False).index[:k]
    e = pd.DataFrame({"dP": dp.loc[top], "basket_AR_pct": 100 * bsk["basket"].reindex(top),
                      "long_AR_pct": 100 * bsk["long"].reindex(top), "short_AR_pct": 100 * bsk["short"].reindex(top)}).sort_index()
    e["signed_basket_AR_pct"] = np.sign(e.dP) * e.basket_AR_pct           # positive = basket moved the way the policy logic says
    return e


if __name__ == "__main__":
    px, r, uni, pm = load()
    mm, ar = market_model(r, uni)
    bsk, ch = basket(ar, uni)
    eb = election_betas(ar, ch, bsk, pm)
    stab, rho, p = stability(ar, pm)
    stab.round(4).to_csv(OUT / "table_beta_stability.csv")
    ev = event_days(bsk, pm)
    ev.round(4).to_csv(OUT / "table_event_days.csv")
    json.dump({"spearman_H1_H2": rho, "p": p, "event_mean_signed_AR_pct": ev.signed_basket_AR_pct.mean(),
               "event_t": ev.signed_basket_AR_pct.mean() / (ev.signed_basket_AR_pct.std(ddof=1) / np.sqrt(len(ev))),
               "event_share_right_sign": float((ev.signed_basket_AR_pct > 0).mean())}, open(OUT / "equity_robustness.json", "w"), indent=1)
    print(f"beta stability H1 vs H2: Spearman {rho:.2f} (p={p:.2f}); event days mean signed AR {ev.signed_basket_AR_pct.mean():.2f}%")
    t = uni.set_index("ticker").join(mm).join(eb)
    t["expected_sign_ok"] = np.sign(t.beta_weekly) == t.sign
    t.round(4).to_csv(OUT / "table_equities.csv")
    sign = uni.groupby("channel").sign.first()
    c = eb[eb.index.str.startswith("[")].copy()
    c["sign"] = [sign.get(i.strip("[]"), np.nan) for i in c.index]
    c.round(4).to_csv(OUT / "table_channels.csv")
    out = bsk.join(ch.add_prefix("ch_")); out["p_dsweep_4pm"] = pm["p"]
    out.to_csv(DATA / "basket_daily.csv")
    pd.set_option("display.width", 220)
    print(c.round(3).to_string())
    print("stocks with expected weekly sign:", int(t.expected_sign_ok.sum()), "/", len(t))
