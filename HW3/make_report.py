"""Assemble the tables of the report (Markdown) from the output CSVs. Narrative sections are written separately in REPORT.md."""
import json, pandas as pd, numpy as np
sp = json.load(open("output/specs.json")); spec = json.load(open("data/var_spec.json"))
rs = pd.read_csv("data/rigobon_sack_2003_table2.csv").set_index("code")
def md(df, floatfmt="{:.3f}"):
    cols = list(df.columns); out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): out.append("| " + " | ".join((floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else ("" if (isinstance(v, float) and pd.isna(v)) else str(v))) for v in r.values) + " |")
    return "\n".join(out)
def fmt_t(b, t): return f"{b:.3g} ({t:.2f})" if pd.notna(b) else ""
L = []
# ---- Table 1 analogue
t1 = pd.read_csv("output/table1_S1_level_q75.csv")
L.append("## Table 1. NLP-selected high-variance war-news days (baseline S1)\n")
L.append("Direction = sign of the escalation-lexicon score of that day's relevant headlines (>0.2 increased risk, <-0.2 decreased). Market columns are the day's changes: 2y yield (bp), S&P 500 (%), Brent front future ($), high-yield OAS (bp).\n")
t1v = t1[["date", "index", "risk_direction", "d2y_bp", "dSPX_pct", "dBrent_usd", "dHY_bp", "headlines"]].copy(); t1v["headlines"] = t1v["headlines"].str.slice(0, 230)
L.append(md(t1v, "{:.2f}"))
# ---- Table 2 analogue for each spec (UST2Y normalisation), paper's 8 variables + selected global ones
for tag, title in [("S1_level_q75", "S1 baseline: full sample, top-25% NLP index, nearest-day L"), ("S4_prewar", "S4 pre-war window (2 Jan - 27 Feb): top-36% NLP index, as in the paper's 17/47 design"), ("S5_warpost", "S5 war and post-war window (2 Mar - 16 Sep)"), ("S6_curated", "S6 hand-curated event days (the paper's 'reading newspapers' approach)"), ("S3_surprise_q75", "S3 surprise index (intensity relative to the trailing 10-day median)"), ("S7_volume_q75", "S7 volume-only index (headline count + Wikipedia text volume), which concentrates on the sustained-war weeks")]:
    t = pd.read_csv(f"output/table2_{tag}_ust2y.csv"); s = sp[tag]
    L.append(f"\n## Table 2 ({tag}). Estimated impact of an increase in war risk, normalised to a 25 bp drop in the two-year Treasury yield\n")
    L.append(f"{title}. Window {s['window'][0]} to {s['window'][1]}; nH = {len(s['H'])}, nL = {len(s['L'])}. Two-year yield variance: L days {s['var2y_L_H'][0]:.5f}, H days {s['var2y_L_H'][1]:.5f}. Absolute t-statistics (heteroskedasticity-robust) in parentheses.\n")
    v = pd.DataFrame({"Variable": t["variable"], "Units": t["units"], "IV with nu1": [fmt_t(b, tt) for b, tt in zip(t["IV nu1"], t["t1"])], "IV with nu2": [fmt_t(b, tt) for b, tt in zip(t["IV nu2"], t["t2"])], "IV with nu3": [fmt_t(b, tt) for b, tt in zip(t["IV nu3"], t["t3"])]})
    L.append(md(v))
    L.append(f"\n### Table 3 ({tag}). Variances and share explained by the war-risk factor\n")
    v3 = t[["variable", "Var L", "Var H", "Pred dVar", "% expl H", "% expl all"]].rename(columns={"variable": "Variable", "Var L": "Var. on L days", "Var H": "Var. on H days", "Pred dVar": "Predicted change in var.", "% expl H": "% explained, H days", "% expl all": "% explained, all days"})
    L.append(md(v3, "{:.4g}"))
# ---- Brent normalisation, baseline
t = pd.read_csv("output/table2_S1_level_q75_brent.csv")
L.append("\n## Table 2b (S1, alternative normalisation). Impact of a war-risk increase that raises the Brent front-month future by $5\n")
L.append(f"Brent variance: L days {sp['S1_level_q75']['varBrent_L_H'][0]:.2f}, H days {sp['S1_level_q75']['varBrent_L_H'][1]:.2f} ($^2).\n")
v = pd.DataFrame({"Variable": t["variable"], "Units": t["units"], "IV with nu1": [fmt_t(b, tt) for b, tt in zip(t["IV nu1"], t["t1"])], "IV with nu2": [fmt_t(b, tt) for b, tt in zip(t["IV nu2"], t["t2"])], "IV with nu3": [fmt_t(b, tt) for b, tt in zip(t["IV nu3"], t["t3"])], "% explained, H days": t["% expl H"].round(1), "% explained, all days": t["% expl all"].round(1)})
L.append(md(v, "{:.1f}"))
# ---- comparison with 2003
L.append("\n## Comparison with Rigobon and Sack (2003), Iraq, Jan 6 - Mar 17 2003 (combined-instrument estimates)\n")
L.append("2003 and the 2026 pre-war window are normalised to a war-risk increase that lowers the two-year yield by 25 bp (the two-year yield variance rises on war days in both). The 2026 full-sample specifications are normalised to a war-risk increase that raises Brent by $5, because the two-year yield variance does not rise on 2026 war days (see Section 3.2). |t| in parentheses.\n")
rows = []
for c in rs.index:
    r = {"Variable": rs.loc[c, "variable"], "Units": rs.loc[c, "units"], "RS 2003 Iraq (-25bp 2y)": fmt_t(rs.loc[c, "RS2003_nu3"], rs.loc[c, "RS2003_t3"])}
    t = pd.read_csv("output/table2_S4_prewar_ust2y.csv").set_index("code"); r["2026 pre-war S4 (-25bp 2y)"] = fmt_t(t.loc[c, "IV nu3"], t.loc[c, "t3"]) if c in t.index else ""
    for tag, lab in [("S1_level_q75", "2026 S1 full sample (+$5 Brent)"), ("S5_warpost", "2026 S5 war/post (+$5 Brent)"), ("S6_curated", "2026 S6 curated (+$5 Brent)")]:
        t = pd.read_csv(f"output/table2_{tag}_brent.csv").set_index("code"); r[lab] = fmt_t(t.loc[c, "IV nu3"], t.loc[c, "t3"]) if c in t.index else ""
    rows.append(r)
L.append(md(pd.DataFrame(rows)))
L.append("\nNote: the 2003 oil variable is the 12-month WTI future; here it is the front-month WTI future. Corporate spreads here are ICE BofA option-adjusted spreads (FRED) rather than Merrill Lynch index spreads to the Treasury curve.")
# ---- summary across specs, all variables, both normalisations
for base, lab in [("ust", "war-risk increase lowering the 2y yield 25 bp"), ("brent", "war-risk increase raising Brent $5")]:
    s = pd.read_csv(f"output/summary_across_specs_{base}.csv")
    L.append(f"\n## Summary across specifications: combined-instrument (nu3) estimates, {lab}; |t| in parentheses\n"); L.append(md(s))
open("output/report_tables.md", "w").write("\n".join(L)); print("wrote output/report_tables.md", len("\n".join(L)), "chars")
