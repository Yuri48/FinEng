"""Step 4: does key-topic sentiment track the market-implied probability of a Democratic sweep, and does either track
the market-adjusted performance of the midterm basket?

Frequency: weeks ending Sunday, 4 Jan - 4 Oct 2026 (40 weeks; prediction markets trade every day, stock returns are
summed Monday-Friday). P_w = last price of the week, mean of Polymarket and Kalshi.
  levels   Pearson and Spearman correlation; slope of z(P) on z(index) with Newey-West(4) t-statistics. Both series
           trend, so levels are reported for completeness; inference rests on changes.
  changes  correlation of weekly changes with a Newey-West(2) t-statistic; lead/lag cross-correlations (k = -4..4,
           positive k = sentiment leads); Granger F-tests with 2 lags in both directions.
Equities: weekly basket abnormal return on dP and d(index), with Brent and 10-year-yield controls.
-> output/table_corr_pm.csv, output/table_leadlag.csv, output/table_granger.csv, output/table_basket_reg.csv,
   output/table_basket_corr.csv, output/fig/*.png
"""
import json, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from config import DATA, OUT, FIG
import matplotlib.dates as mdates
from plotstyle import plt, save, SERIES, INK, INK2, MUTED, NEUTRAL, BLUE, ORANGE


def months(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
from sentiment_index import KEY_ISSUES

warnings.filterwarnings("ignore")
W0, W1 = "2026-01-04", "2026-10-04"
SHORT = {"Economy & cost of living": "Economy", "Democracy & election integrity": "Democracy", "Iran war & foreign policy": "Iran war",
         "Corruption & accountability": "Corruption", "Immigration & ICE": "Immigration", "Health care": "Health care", "AI, tech & data centres": "AI/data centres",
         "Tariffs & trade": "Tariffs"}


def panel():
    pm = pd.read_csv(DATA / "pm_daily.csv", index_col=0, parse_dates=True).ffill(limit=5)   # Polymarket's daily feed has a few gaps
    pm["p"] = pm[["poly_DD", "kalshi_DD"]].mean(axis=1)
    P = pm[["p", "poly_DD", "kalshi_DD", "kalshi_house_D", "kalshi_senate_D"]].resample("W-SUN").last()
    K = pd.read_csv(DATA / "ktsi_weekly.csv", index_col=0, parse_dates=True)
    iw = pd.read_csv(DATA / "issue_week.csv", header=[0, 1], index_col=0, parse_dates=True)
    issue = pd.concat({f"{m}|{SHORT[k]}": iw[(m, k)] for m in ("tone", "pro_dem") for k in KEY_ISSUES if (m, k) in iw}, axis=1)
    b = pd.read_csv(DATA / "basket_daily.csv", index_col=0, parse_dates=True)
    B = b.drop(columns=["p_dsweep_4pm"]).resample("W-SUN").sum(min_count=2)
    px = pd.read_csv(DATA / "prices.csv", index_col=0, parse_dates=True)
    wk = px[["BZ=F", "SPY", "^TNX"]].resample("W-SUN").last()
    ctrl = pd.DataFrame({"brent_ret": wk["BZ=F"].pct_change(), "spy_ret": wk["SPY"].pct_change(), "d10y": wk["^TNX"].diff()})
    df = P.join(K).join(issue).join(B).join(ctrl).loc[W0:W1]
    return df


def hac_t(y, x, lags):
    d = pd.concat([y, x], axis=1).dropna()
    f = sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return f.params.iloc[1], f.tvalues.iloc[1], f.pvalues.iloc[1], int(f.nobs)


def corr_row(x, y):
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    z = (d - d.mean()) / d.std()
    _, t_l, p_l, n = hac_t(z.y, z.x, 4)
    dd = d.diff().dropna()
    _, t_c, p_c, n_c = hac_t(dd.y, dd.x, 2)
    return {"r_level": d.x.corr(d.y), "rho_level": stats.spearmanr(d.x, d.y)[0], "t_level_NW": t_l, "p_level_NW": p_l, "n": n,
            "r_change": dd.x.corr(dd.y), "t_change_NW": t_c, "p_change_NW": p_c, "n_change": n_c}


def granger(x, y, lag=2):
    """p-values: x Granger-causes y, and y Granger-causes x (changes)."""
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).diff().dropna()
    p_xy = grangercausalitytests(d[["y", "x"]], maxlag=[lag])[lag][0]["ssr_ftest"][1]
    p_yx = grangercausalitytests(d[["x", "y"]], maxlag=[lag])[lag][0]["ssr_ftest"][1]
    return p_xy, p_yx


def leadlag(x, y, K=4):
    dx, dy = x.diff(), y.diff()
    return {k: dy.corr(dx.shift(k)) for k in range(-K, K + 1)}          # k > 0: x leads y by k weeks


def sentiment_vs_pm(df):
    idx_cols = ["ktsi_tone", "ktsi_prodem", "ktsi_vader", "ktsi_tone_eqw", "ktsi_prodem_eqw"] + \
               [c for c in df.columns if c.startswith(("tone|", "pro_dem|"))]
    rows, ll, gr = [], {}, []
    for target in ("p", "poly_DD", "kalshi_DD"):
        for c in idx_cols:
            r = corr_row(df[c], df[target]); r.update({"index": c, "target": target}); rows.append(r)
    for c in ["ktsi_tone", "ktsi_prodem", "ktsi_vader"] + [c for c in df.columns if c.startswith(("tone|", "pro_dem|"))]:
        ll[c] = leadlag(df[c], df["p"])
        p1, p2 = granger(df[c], df["p"]); gr.append({"index": c, "p_index_causes_P": p1, "p_P_causes_index": p2})
    t = pd.DataFrame(rows).set_index(["target", "index"])
    t.round(4).to_csv(OUT / "table_corr_pm.csv")
    pd.DataFrame(ll).T.round(3).to_csv(OUT / "table_leadlag.csv")
    pd.DataFrame(gr).set_index("index").round(4).to_csv(OUT / "table_granger.csv")
    return t


def basket_tests(df):
    out = []
    df = df.copy()
    df["car_basket"] = df.basket.cumsum(); df["dp"] = df.p.diff()
    df["d_tone"] = df.ktsi_tone.diff(); df["d_prodem"] = df.ktsi_prodem.diff()
    specs = {"(1) dP": ["dp"], "(2) d tone index": ["d_tone"], "(3) d pro-Dem index": ["d_prodem"],
             "(4) dP + controls": ["dp", "brent_ret", "d10y"], "(5) d pro-Dem + controls": ["d_prodem", "brent_ret", "d10y"],
             "(6) all": ["dp", "d_tone", "d_prodem", "brent_ret", "d10y"]}
    for y in ("basket", "long", "short"):
        for name, xs in specs.items():
            d = df[[y] + xs].dropna()
            f = sm.OLS(100 * d[y], sm.add_constant(d[xs])).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
            for x in xs:
                out.append({"y": y, "spec": name, "x": x, "coef": f.params[x], "t": f.tvalues[x], "p": f.pvalues[x],
                            "r2": f.rsquared, "n": int(f.nobs)})
    reg = pd.DataFrame(out)
    reg.round(4).to_csv(OUT / "table_basket_reg.csv", index=False)
    rows = []
    for c in ["ktsi_tone", "ktsi_prodem", "p"]:
        r = corr_row(df[c], df.car_basket); r["index"] = c; rows.append(r)
    for ch in [c for c in df.columns if c.startswith("ch_")]:
        car = df[ch].cumsum()
        for c in ["ktsi_prodem", "p"]:
            r = corr_row(df[c], car); r["index"] = f"{c} ~ {ch[3:]}"; rows.append(r)
    bc = pd.DataFrame(rows).set_index("index")
    bc.round(4).to_csv(OUT / "table_basket_corr.csv")
    return reg, bc, df


# ----------------------------------------------------------------------------------------------- figures
def fig_salience():
    s = pd.read_csv(OUT / "table_salience.csv", index_col=0)
    s = s[s.index != "Other / noise"].sort_values("voter_share")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(s)); h = 0.38
    ax.barh(y + h / 2, 100 * s.voter_share, h * 0.92, color=BLUE, label="Social media (4 platforms, equal weight)")
    ax.barh(y - h / 2, 100 * s.news_share, h * 0.92, color=ORANGE, label="News headlines")
    ax.set_yticks(y, s.index); ax.grid(axis="y", visible=False)
    for i, v in enumerate(100 * s.voter_share):
        ax.text(v + 0.3, i + h / 2, f"{v:.0f}%", va="center", fontsize=7, color=INK2)
    ax.set_xlabel("Share of issue-related documents (%)")
    ax.set_title("Figure 1. What the midterm conversation is about, Jan-Oct 2026")
    ax.legend(loc="lower right")
    save(fig, FIG / "fig1_salience.png")


def fig_issue_salience_time():
    from sentiment_index import load, NON_ISSUES
    s = load()
    s = s[(s.source != "news2") & s.issue.notna() & ~s.issue.isin(NON_ISSUES)]
    s["grp"] = np.where(s.kind == "social", "social", "news")
    sh = s.groupby(["week", "grp", "issue"]).size() / s.groupby(["week", "grp"]).size()
    sh = sh.unstack("issue").fillna(0).loc[pd.IndexSlice[:, :], :]
    fig, axes = plt.subplots(5, 2, figsize=(7.4, 9.2), sharex=True)
    for ax, k in zip(axes.flat, KEY_ISSUES):
        for grp, col, lab in (("social", BLUE, "Social media"), ("news", ORANGE, "News headlines")):
            x = sh.xs(grp, level="grp")[k].loc[W0:W1] if k in sh else None
            if x is not None: ax.plot(x.index, 100 * x.rolling(2, min_periods=1).mean(), color=col, lw=1.3, label=lab)
        ax.set_title(k, fontsize=9); ax.set_ylim(bottom=0)
    tot = s.groupby(["week", "grp"]).size().unstack("grp").loc[W0:W1]
    for a in axes.flat[len(KEY_ISSUES) + 1:]: a.set_visible(False)
    ax = axes.flat[len(KEY_ISSUES)]
    ax.plot(tot.index, tot.social, color=BLUE, lw=1.3); ax.plot(tot.index, tot.news, color=ORANGE, lw=1.3)
    ax.set_title("Issue documents per week (count)", fontsize=9); ax.set_ylim(bottom=0)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Figure 2. Weekly share of issue-related documents by key issue (%, 2-week average)", x=0.02, ha="left",
                 fontweight="bold", fontsize=10)
    for a in axes.flat: months(a); a.tick_params(axis="x", labelsize=7); a.xaxis.set_tick_params(labelbottom=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, FIG / "fig2_issue_salience_time.png")


def fig_issue_sentiment(df):
    fig, axes = plt.subplots(5, 2, figsize=(7.4, 9.6), sharex=True)
    for ax, k in zip(axes.flat, KEY_ISSUES):
        sk = SHORT[k]
        for c, col, lab in ((f"tone|{sk}", BLUE, "Tone (RoBERTa)"), (f"pro_dem|{sk}", ORANGE, "Pro-Democratic (ABSA)")):
            if c in df:
                z = (df[c] - df[c].mean()) / df[c].std()
                ax.plot(df.index, z, color=col, lw=1.3, label=lab)
        ax.axhline(0, color=NEUTRAL, lw=0.6); ax.set_title(k, fontsize=9)
    for a in axes.flat[len(KEY_ISSUES) + 1:]: a.set_visible(False)
    ax = axes.flat[len(KEY_ISSUES)]
    z = (df.p - df.p.mean()) / df.p.std()
    ax.plot(df.index, z, color=INK, lw=1.4, label="P(Democratic sweep)"); ax.axhline(0, color=NEUTRAL, lw=0.6)
    ax.set_title("P(Democratic sweep), for reference", fontsize=9)
    h1, l1 = axes.flat[0].get_legend_handles_labels()
    fig.legend(h1 + ax.get_legend_handles_labels()[0], l1 + ["P(Democratic sweep)"], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Figure 3. Weekly sentiment by key issue (z-scores, source-demeaned)", x=0.02, ha="left", fontweight="bold", fontsize=10)
    for a in axes.flat: months(a); a.tick_params(axis="x", rotation=0, labelsize=7); a.xaxis.set_tick_params(labelbottom=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, FIG / "fig3_issue_sentiment.png")


EVENTS = {"2026-02-28": "Iran war\nbegins", "2026-05-07": "Va. court voids\nDem map", "2026-09-15": "GOP breaks with\nTrump on Iran"}


def fig_index_vs_pm(df):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    a1.plot(df.index, 100 * df.poly_DD, color=BLUE, label="Polymarket: D Senate + D House")
    a1.plot(df.index, 100 * df.kalshi_DD, color=ORANGE, label="Kalshi: Democrats sweep")
    a1.set_ylabel("Probability (%)"); a1.legend(loc="lower right")
    a1.set_title("Figure 4. Prediction-market P(Democratic sweep) and the Key-Topic Sentiment Index (weekly)")
    for c, col, lab in (("ktsi_tone", BLUE, "KTSI tone (higher = more positive)"), ("ktsi_prodem", ORANGE, "KTSI pro-Democratic")):
        z = (df[c] - df[c].mean()) / df[c].std()
        a2.plot(df.index, z, color=col, label=lab)
    a2.axhline(0, color=NEUTRAL, lw=0.6); a2.set_ylabel("z-score")
    a2.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    for d, lab in EVENTS.items():
        for a in (a1, a2): a.axvline(pd.Timestamp(d), color=MUTED, lw=0.6)
        a1.text(pd.Timestamp(d), a1.get_ylim()[1], " " + lab, va="top", fontsize=7, color=INK2)
    months(a2)
    fig.tight_layout()
    save(fig, FIG / "fig4_index_vs_pm.png")


def fig_changes(df):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    dp = 100 * df.p.diff()
    for ax, c, col, lab in ((axes[0], "ktsi_tone", BLUE, "Change in KTSI tone"), (axes[1], "ktsi_prodem", ORANGE, "Change in KTSI pro-Democratic")):
        dx = df[c].diff(); d = pd.concat([dx, dp], axis=1).dropna()
        ax.scatter(d.iloc[:, 0], d.iloc[:, 1], s=16, color=col, edgecolor="white", linewidth=0.6, zorder=3)
        b = np.polyfit(d.iloc[:, 0], d.iloc[:, 1], 1); xs = np.linspace(d.iloc[:, 0].min(), d.iloc[:, 0].max(), 10)
        ax.plot(xs, np.polyval(b, xs), color=INK2, lw=1)
        ax.set_xlabel(lab); ax.text(0.03, 0.95, f"r = {d.corr().iloc[0, 1]:.2f}", transform=ax.transAxes, va="top", color=INK2)
    axes[0].set_ylabel("Change in P(sweep), points")
    fig.suptitle("Figure 5. Weekly changes: sentiment index vs P(Democratic sweep)", x=0.02, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    save(fig, FIG / "fig5_changes.png")


def fig_leadlag():
    ll = pd.read_csv(OUT / "table_leadlag.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ks = np.array([int(float(k)) for k in ll.columns]); w = 0.38
    ax.bar(ks - w / 2, ll.loc["ktsi_tone"], w * 0.92, color=BLUE, label="KTSI tone")
    ax.bar(ks + w / 2, ll.loc["ktsi_prodem"], w * 0.92, color=ORANGE, label="KTSI pro-Democratic")
    band = 1.96 / np.sqrt(38)
    ax.axhspan(-band, band, color="#f0efec", zorder=0); ax.axhline(0, color=NEUTRAL, lw=0.6)
    ax.set_xticks(ks, [f"{k:+d}" for k in ks]); ax.set_xlabel("k (weeks); k > 0: sentiment change leads the P(sweep) change")
    ax.set_ylabel("Correlation"); ax.legend(loc="upper left")
    ax.set_title("Figure 6. Lead-lag correlations of weekly changes (shaded: ±1.96/√n)")
    save(fig, FIG / "fig6_leadlag.png")


def fig_basket(df):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 5.4), sharex=True)
    a1.plot(df.index, 100 * df.p, color=INK, label="P(Democratic sweep), mean of Polymarket and Kalshi")
    a1.set_ylabel("Probability (%)"); a1.legend(loc="upper left")
    a1.set_title("Figure 7. The midterm basket: cumulative market-adjusted (CAPM) abnormal return")
    for c, col, lab in (("basket", INK, "Basket = Dem-sweep winners minus GOP-hold winners"), ("long", BLUE, "Dem-sweep winners leg"),
                        ("short", ORANGE, "GOP-hold winners leg")):
        a2.plot(df.index, 100 * df[c].cumsum(), color=col, label=lab, lw=1.8 if c == "basket" else 1.3)
    a2.axhline(0, color=NEUTRAL, lw=0.6); a2.set_ylabel("Cumulative abnormal return (%)")
    a2.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    for d, lab in EVENTS.items():
        for a in (a1, a2): a.axvline(pd.Timestamp(d), color=MUTED, lw=0.6)
        a1.text(pd.Timestamp(d), a1.get_ylim()[1], " " + lab, va="top", fontsize=7, color=INK2)
    months(a2)
    fig.tight_layout()
    save(fig, FIG / "fig7_basket.png")


def fig_channels():
    c = pd.read_csv(OUT / "table_channels.csv", index_col=0)
    c = c[~c.index.str.contains("BASKET|leg")].copy()
    c.index = c.index.str.strip("[]"); c = c.sort_values("beta_weekly")
    se = c.beta_weekly / c.t_weekly
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = np.arange(len(c))
    cols = [BLUE if s > 0 else ORANGE for s in c.sign]
    ax.errorbar(c.beta_weekly, y, xerr=1.96 * se, fmt="none", ecolor=NEUTRAL, elinewidth=1.2, capsize=0)
    ax.scatter(c.beta_weekly, y, s=40, c=cols, zorder=3, edgecolor="white", linewidth=1)
    ax.axvline(0, color=INK2, lw=0.7); ax.set_yticks(y, c.index); ax.grid(axis="y", visible=False)
    ax.scatter([], [], c=BLUE, s=40, label="Expected to gain from a Democratic sweep")
    ax.scatter([], [], c=ORANGE, s=40, label="Expected to lose from a Democratic sweep")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlabel("Weekly abnormal return per unit change in P(sweep) (0.10 = 1% per 10 points), 95% CI")
    ax.set_title("Figure 8. Election betas by policy channel")
    save(fig, FIG / "fig8_channel_betas.png")


if __name__ == "__main__":
    df = panel()
    t = sentiment_vs_pm(df)
    pd.set_option("display.width", 240)
    print(t.loc["p"].round(3).to_string())
    print(pd.read_csv(OUT / "table_granger.csv", index_col=0).round(3).to_string())
    print(pd.read_csv(OUT / "table_leadlag.csv", index_col=0).round(2).to_string())
    reg, bc, df2 = basket_tests(df)
    print(reg[reg.y == "basket"].round(3).to_string())
    print(bc.round(3).to_string())
    df2.to_csv(DATA / "weekly_panel.csv")
    fig_salience(); fig_issue_salience_time(); fig_issue_sentiment(df); fig_index_vs_pm(df); fig_changes(df); fig_leadlag(); fig_basket(df); fig_channels()
