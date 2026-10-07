"""Fill the generated tables into REPORT.md (between <!-- BEGIN name --> / <!-- END name --> markers), render a
self-contained REPORT.html with the figures embedded, and print report.pdf with headless Chrome.

    python build_report.py            # tables + HTML + PDF
"""
import base64, html, json, os, re, subprocess
import numpy as np
import pandas as pd
from config import OUT, DATA

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def md(df, fmt="{:.2f}", index=False):
    if index: df = df.reset_index()
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(cols))) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else fmt.format(v))
            else:
                cells.append(str(v).replace("|", "/"))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def star(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def tables():
    T = {}
    # Table 1: corpus
    c = pd.read_parquet(DATA / "corpus.parquet")
    s = pd.read_parquet(DATA / "doc_sentiment.parquet")
    n2 = s[s.source == "news2"]
    rows = []
    desc = {"news": "Google News RSS, 2 issue-free election queries a day", "bluesky": "Top 100 English posts a day matching 'midterms'",
            "mastodon": "mastodon.social tag timelines: #midterms, #midterms2026, #midterm, #election2026, #2026midterms",
            "reddit": "Comments mentioning 'midterm' in 8 political subreddits (sampled windows, Arctic Shift archive)",
            "lemmy": "lemmy.world comments and posts mentioning 'midterm'"}
    for src in ["news", "bluesky", "mastodon", "reddit", "lemmy"]:
        d = c[c.source == src]
        rows.append({"Source": src.capitalize() if src != "news" else "News headlines (discovery)", "What": desc[src], "Documents": f"{len(d):,}",
                     "Days covered": d.date.nunique(), "Median length (chars)": int(d.text.str.len().median())})
    rows.append({"Source": "News headlines (issue queries)", "What": "Google News RSS, 7 issue-anchored election queries a day (stage 2)",
                 "Documents": f"{len(n2):,}", "Days covered": n2.date.nunique(), "Median length (chars)": ""})
    rows.append({"Source": "Total", "What": "", "Documents": f"{len(c) + len(n2):,}", "Days covered": "", "Median length (chars)": ""})
    T["corpus"] = md(pd.DataFrame(rows))
    # Table 2: salience
    sal = pd.read_csv(OUT / "table_salience.csv", index_col=0)
    sal = sal[sal.index != "Other / noise"]
    t = pd.DataFrame({"Issue": sal.index, "Social share (%)": 100 * sal.voter_share, "News share (%)": 100 * sal.news_share,
                      "Rank (social)": sal.rank_voter, "Rank (news)": sal.rank_news, "BGSU poll, Sep 2026": sal.poll_BGSU_Sep2026.fillna("")})
    for c_ in ["bluesky", "mastodon", "reddit", "lemmy"]:
        if f"share_{c_}" in sal: t.insert(3 + ["bluesky", "mastodon", "reddit", "lemmy"].index(c_), c_.capitalize() + " (%)", 100 * sal[f"share_{c_}"].values)
    T["salience"] = md(t, "{:.1f}")
    # Table 3: validation
    v = pd.read_csv(OUT / "table_validation.csv")
    v1 = v[v.target == "tone"][["measure", "n", "spearman", "accuracy_3class", "macro_f1", "majority_baseline"]]
    v1.columns = ["Tone measure", "n", "Spearman ρ", "3-class accuracy", "Macro-F1", "Majority-class baseline"]
    v2 = v[v.target == "partisan"][["measure", "n", "coverage", "spearman", "n_nonzero", "sign_accuracy_nonzero"]]
    v2.columns = ["Partisan measure", "n scored", "Coverage", "Spearman ρ", "n labelled ±1", "Sign accuracy"]
    T["validation"] = md(v1, "{:.2f}") + "\n\n" + md(v2, "{:.2f}")
    # Table 4: correlations with P(sweep)
    cp = pd.read_csv(OUT / "table_corr_pm.csv")
    cp = cp[cp.target == "p"]
    name = {"ktsi_tone": "KTSI tone (RoBERTa, salience-weighted)", "ktsi_prodem": "KTSI pro-Democratic (ABSA, salience-weighted)",
            "ktsi_vader": "KTSI tone (VADER)", "ktsi_tone_eqw": "KTSI tone, equal weights", "ktsi_prodem_eqw": "KTSI pro-Democratic, equal weights"}
    cp["Index"] = cp["index"].map(lambda x: name.get(x, x.replace("tone|", "Tone: ").replace("pro_dem|", "Pro-Dem: ")))
    t = pd.DataFrame({"Index": cp.Index, "r (levels)": cp.r_level, "ρ (levels)": cp.rho_level,
                      "NW t (levels)": [f"{a:.2f}{star(p)}" for a, p in zip(cp.t_level_NW, cp.p_level_NW)],
                      "r (weekly changes)": cp.r_change, "NW t (changes)": [f"{a:.2f}{star(p)}" for a, p in zip(cp.t_change_NW, cp.p_change_NW)],
                      "Weeks": cp.n})
    T["corr_pm"] = md(t, "{:.2f}")
    # Table 5: lead-lag and Granger for composites + issues
    ll = pd.read_csv(OUT / "table_leadlag.csv", index_col=0)
    gr = pd.read_csv(OUT / "table_granger.csv", index_col=0)
    t = ll.copy(); t.columns = [f"k={int(float(k)):+d}" for k in t.columns]
    t["Granger: index → P (p)"] = gr.p_index_causes_P; t["Granger: P → index (p)"] = gr.p_P_causes_index
    t.index = [name.get(x, x.replace("tone|", "Tone: ").replace("pro_dem|", "Pro-Dem: ")) for x in t.index]
    T["leadlag"] = md(t, "{:.2f}", index=True).replace("| index |", "| Index |", 1)
    # Table 6: channels
    ch = pd.read_csv(OUT / "table_channels.csv", index_col=0)
    eq = pd.read_csv(OUT / "table_equities.csv", index_col=0)
    members = eq.groupby("channel").apply(lambda d: ", ".join(d.index))
    chs = ch[~ch.index.str.contains("BASKET|leg")].copy(); chs.index = chs.index.str.strip("[]")
    t = pd.DataFrame({"Channel": chs.index, "Stocks": [members.get(i, "") for i in chs.index],
                      "Expected sign": ["+" if s > 0 else "−" for s in chs.sign], "CAR 2026 (%)": chs.car_2026_pct,
                      "β daily (t)": [f"{b:.2f} ({tt:.2f})" for b, tt in zip(chs.beta_daily, chs.t_daily)],
                      "β weekly (t)": [f"{b:.2f} ({tt:.2f})" for b, tt in zip(chs.beta_weekly, chs.t_weekly)],
                      "Sign as expected": ["yes" if np.sign(b) == s else "no" for b, s in zip(chs.beta_weekly, chs.sign)]})
    agg = ch[ch.index.str.contains("BASKET|leg")]
    t2 = pd.DataFrame({"Channel": agg.index.str.strip("[]"), "Stocks": "", "Expected sign": ["+", "+", "−"][:len(agg)],
                       "CAR 2026 (%)": agg.car_2026_pct, "β daily (t)": [f"{b:.2f} ({tt:.2f})" for b, tt in zip(agg.beta_daily, agg.t_daily)],
                       "β weekly (t)": [f"{b:.2f} ({tt:.2f})" for b, tt in zip(agg.beta_weekly, agg.t_weekly)], "Sign as expected": ""})
    T["channels"] = md(pd.concat([t, t2]), "{:.1f}")
    # Table 7: basket regressions (basket only)
    br = pd.read_csv(OUT / "table_basket_reg.csv")
    br = br[br.y == "basket"]
    lab = {"dp": "ΔP(sweep)", "d_tone": "ΔKTSI tone", "d_prodem": "ΔKTSI pro-Dem", "brent_ret": "Brent return", "d10y": "Δ10y yield"}
    piv = br.assign(cell=[f"{c:.2f} ({t:.2f}){star(p)}" for c, t, p in zip(br.coef, br.t, br.p)]).pivot(index="x", columns="spec", values="cell")
    piv = piv.reindex([k for k in lab if k in piv.index]); piv.index = [lab[i] for i in piv.index]
    r2 = br.groupby("spec").r2.first(); nn = br.groupby("spec").n.first()
    piv.loc["R²"] = [f"{r2[c]:.3f}" for c in piv.columns]; piv.loc["Weeks"] = [str(nn[c]) for c in piv.columns]
    T["basket_reg"] = md(piv.fillna(""), index=True).replace("| index |", "| Regressor |", 1)
    bc = pd.read_csv(OUT / "table_basket_corr.csv", index_col=0)
    keep = bc.loc[["ktsi_tone", "ktsi_prodem", "p"]]
    keep.index = ["KTSI tone", "KTSI pro-Democratic", "P(Democratic sweep)"]
    t = pd.DataFrame({"Series correlated with the basket": keep.index, "r (levels: CAR)": keep.r_level,
                      "NW t (levels)": [f"{a:.2f}{star(p)}" for a, p in zip(keep.t_level_NW, keep.p_level_NW)],
                      "r (weekly changes: AR)": keep.r_change, "NW t (changes)": [f"{a:.2f}{star(p)}" for a, p in zip(keep.t_change_NW, keep.p_change_NW)]})
    T["basket_corr"] = md(t, "{:.2f}")
    # Appendix A: topic -> issue mapping
    ti = pd.read_csv(OUT / "topic_info.csv")
    lab = json.load(open(DATA / "topic_labels.json"))
    ti["Label"] = ti.topic.map(lambda x: lab[str(x)]["label"]); ti["Issue"] = ti.topic.map(lambda x: lab[str(x)]["issue"])
    ti["Keywords (c-TF-IDF)"] = ti.words.str.split(", ").str[:7].str.join(", ")
    ti = ti.sort_values(["Issue", "n"], ascending=[True, False])
    a = pd.DataFrame({"Topic": ti.topic, "Docs": ti.n, "Social share": ti.social_share_of_topic, "Issue": ti.Issue, "Label (LLM)": ti.Label,
                      "Keywords (c-TF-IDF)": ti["Keywords (c-TF-IDF)"]})
    T["topics"] = md(a, "{:.2f}")
    # Appendix B: equities
    e = pd.read_csv(OUT / "table_equities.csv", index_col=0)
    st = pd.read_csv(OUT / "table_beta_stability.csv", index_col=0)
    e = e.join(st)
    b = pd.DataFrame({"Ticker": e.index, "Channel": e.channel, "Sign": ["+" if s > 0 else "−" for s in e.sign], "CAPM β (2025)": e.beta_2025,
                      "CAR 2026 (%)": e.car_2026_pct, "Election β weekly (t)": [f"{x:.2f} ({y:.2f})" for x, y in zip(e.beta_weekly, e.t_weekly)],
                      "β Jan-May": e.beta_H1, "β Jun-Oct": e.beta_H2})
    T["equities_detail"] = md(b, "{:.2f}")
    ev = pd.read_csv(OUT / "table_event_days.csv")
    ev = pd.DataFrame({"Date": ev.date, "ΔP(sweep), points": 100 * ev.dP, "Basket AR (%)": ev.basket_AR_pct, "Dem-winner leg (%)": ev.long_AR_pct,
                       "GOP-winner leg (%)": ev.short_AR_pct})
    T["event_days"] = md(ev, "{:.2f}")
    return T


def fill(md_text, T):
    for k, v in T.items():
        md_text = re.sub(rf"(<!-- BEGIN {k} -->)(.*?)(<!-- END {k} -->)", lambda m: f"{m.group(1)}\n{v}\n{m.group(3)}", md_text, flags=re.S)
    return md_text


def md_to_html(md_text):
    out = []; lines = md_text.split("\n"); i = 0

    def embed(p):
        if os.path.exists(p):
            return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()
        return p

    def inline(t):
        t = html.escape(t, quote=False)
        t = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", lambda m: f'<img alt="{m.group(1)}" src="{embed(m.group(2))}">', t)
        t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
        return re.sub(r"`([^`]+)`", r"<code>\1</code>", t)

    while i < len(lines):
        l = lines[i]
        if l.strip().startswith("<!--"):
            i += 1; continue
        if l.strip() == "---":
            out.append("<hr>"); i += 1; continue
        if re.match(r"^!\[[^\]]*\]\([^)]+\)\s*$", l.strip()):
            out.append(f'<p class="fig">{inline(l.strip())}</p>'); i += 1; continue
        if l.startswith("|") and i + 1 < len(lines) and re.match(r"^\|\s*:?-", lines[i + 1]):
            hdr = [c.strip() for c in l.strip().strip("|").split("|")]; i += 2; rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
            num = lambda c: bool(re.match(r"^[−\-+]?[\d.,]+%?(\s*\([−\-\d.]+\))?\**$", c))
            out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in hdr) + "</tr></thead><tbody>" +
                       "".join("<tr>" + "".join(f'<td class="{"n" if num(c) else "t"}">{inline(c)}</td>' for c in r) + "</tr>" for r in rows) +
                       "</tbody></table>")
            continue
        if l.lstrip().startswith("<") :
            blk = []
            while i < len(lines) and lines[i].strip(): blk.append(lines[i]); i += 1
            out.append("\n".join(blk)); continue
        m = re.match(r"^(#{1,4})\s+(.*)", l)
        if m:
            out.append(f"<h{len(m.group(1))}>{inline(m.group(2))}</h{len(m.group(1))}>"); i += 1; continue
        if re.match(r"^\s*[-*]\s+", l):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                item = re.sub(r"^\s*[-*]\s+", "", lines[i]); i += 1
                while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() and not re.match(r"^\s*[-*]\s+", lines[i]):
                    item += " " + lines[i].strip(); i += 1
                items.append(item)
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ul>"); continue
        if re.match(r"^\s*\d+\.\s+", l):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                item = re.sub(r"^\s*\d+\.\s+", "", lines[i]); i += 1
                while i < len(lines) and lines[i].startswith("   ") and lines[i].strip():
                    item += " " + lines[i].strip(); i += 1
                items.append(item)
            out.append("<ol>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ol>"); continue
        if l.strip() == "":
            i += 1; continue
        para = [l]; i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("|", "#", "<")) and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
            para.append(lines[i]); i += 1
        out.append(f"<p>{inline(' '.join(para))}</p>")
    return "\n".join(out)


CSS = """body{font-family:Georgia,serif;max-width:1050px;margin:30px auto;padding:0 20px;line-height:1.45;color:#1b1b1b;background:#fff}
h1{font-size:1.6em;margin-bottom:.2em}h2{border-bottom:1px solid #ccc;margin-top:1.5em;font-size:1.25em}h3{font-size:1.05em}
table{border-collapse:collapse;font-size:0.8em;margin:10px 0;font-family:Helvetica,Arial,sans-serif;width:100%}
th,td{border:1px solid #ddd;padding:3px 6px;vertical-align:top}th{background:#f3f3f1;text-align:left}td.n{text-align:right;white-space:nowrap}
img{max-width:100%;margin:6px 0}code{background:#f5f5f5;padding:1px 3px;font-size:.9em}p{text-align:justify}p.meta{text-align:left;line-height:1.7}
p.caption{font-size:.85em;color:#555;margin-top:-4px}p.fig{text-align:center;margin:10px 0}
hr{border:0;border-top:1px solid #ccc;margin:18px 0}
@page{size:A4;margin:13mm}@media print{body{max-width:none;margin:0;font-size:10.5pt}table{font-size:7.4pt}td,th{padding:2px 4px}
h2{page-break-after:avoid}img{page-break-inside:avoid}tr{page-break-inside:avoid}}"""


if __name__ == "__main__":
    T = tables()
    text = fill(open("REPORT.md").read(), T)
    open("REPORT.md", "w").write(text)
    body = md_to_html(text)
    open("REPORT.html", "w").write(f"<!doctype html><html><head><meta charset='utf-8'><title>Midterms 2026</title><style>{CSS}</style></head><body>{body}</body></html>")
    if os.path.exists(CHROME):
        prof = "/tmp/hw5-chrome-profile"           # own profile, so headless Chrome does not queue behind a running Chrome
        subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={prof}",
                        f"--print-to-pdf={os.path.abspath('report.pdf')}",
                        "file://" + os.path.abspath("REPORT.html")], capture_output=True, timeout=180)
    print("REPORT.html", os.path.getsize("REPORT.html") // 1024, "KB;", "report.pdf", os.path.getsize("report.pdf") // 1024 if os.path.exists("report.pdf") else "-", "KB")
