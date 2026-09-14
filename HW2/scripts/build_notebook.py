"""Write analysis.ipynb from source. Run, then execute it with nbconvert to save outputs."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Assignment 2: Evaluating the Impact of FOMC Communications on Asset Prices

FRE-GY 7871 A · NLP and the Investment Process · Fall 2026 · Yuri Moghaddam Nasrollahi (ym3414)

This notebook runs on the cached corpus, tone scores and market panel produced by `scripts/01–03`
(none of which are committed). It produces every exhibit in the report, in the order of the assignment sheet:
Table 1 (documents), Figure 1 (tone over time), Table 2 (Warsh-era releases and market changes),
Table 3 (regressions with the 3-month bill control), and the inputs to the forecast.""")

code("""import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
from src import config as C
from src.collect import load_corpus
from src.lexicon import n_patterns, HAWKISH, DOVISH
from src.market import load_panel, attach_market, CHANGE_COLS, CHANGE_LABELS
from src.regress import add_tone_changes, run_table, format_table, stars
from src import exhibits as X
pd.set_option('display.width', 160, 'display.max_columns', 30, 'display.max_colwidth', 50)

corpus = load_corpus()
scores = pd.read_csv(C.INTERIM / 'scores.csv', parse_dates=['date'])
panel = load_panel()
print(len(corpus), 'documents;', scores.shape, 'score rows; panel', panel.index.min().date(), '→', panel.index.max().date())""")

md("## 1. The corpus (Table 1)")
code("""t1 = X.table1(corpus)
t1""")
code("""# Warsh's post-meeting statements are much shorter than Powell's
corpus[corpus.subtype == 'statement'].groupby('chair').n_words.describe()[['count', 'mean', 'min', 'max']]""")

md("""## 2. Tone scores

Three scores per document:

* **Word list** (`lex_net`): 56 hawkish and 55 dovish phrase patterns built for monetary-policy language,
  in four themes (inflation, activity, labor, policy) with negation handling. Score = (H − D)/(H + D) ∈ [−1, 1].
* **FinBERT sentiment** (`fb_sent`): mean over sentences of P(positive) − P(negative) from `ProsusAI/finbert`.
* **FinBERT similarity** (`fb_sim`): mean over sentences of cos(sentence, hawkish anchors) − cos(sentence, dovish anchors),
  where the anchors are key sentences such as *"Interest rates will rise"* / *"Interest rates will fall"*.""")
code("""print(n_patterns())
print('Hawkish policy phrases, e.g.:', HAWKISH['policy'][:4])
print('Dovish inflation phrases, e.g.:', DOVISH['inflation'][:3])""")
code("""scores[C.TONE_SCORES + ['n_sent']].describe().round(3)""")
code("""# correlation between the three measures, by document type
for dt in ['statement', 'minutes', 'speech']:
    print(dt); print(scores[scores.doc_type == dt][C.TONE_SCORES].corr().round(2)); print()""")

md("### Figure 1. Tone over time by document type")
code("""path = X.figure1(scores)
display(Image(path))""")

md("### Powell vs Warsh")
code("""by_chair = X.summary_by_chair(scores)
by_chair.round(3)""")
code("""# Welch t-tests / rank tests: is the Warsh-era mean different from Powell's, by document type?
from scipy import stats
rows = []
for dt in ['statement', 'minutes', 'speech']:
    for s in C.TONE_SCORES:
        p = scores[(scores.doc_type == dt) & (scores.chair == 'Powell')][s].dropna()
        w = scores[(scores.doc_type == dt) & (scores.chair == 'Warsh')][s].dropna()
        pct = (p < w.mean()).mean()   # where Warsh's mean sits in Powell's distribution
        rows.append(dict(doc_type=dt, score=s, powell_mean=p.mean(), warsh_mean=w.mean(), n_w=len(w),
                         warsh_pctile_in_powell=pct, mwu_p=stats.mannwhitneyu(p, w).pvalue if len(w) > 1 else np.nan))
cmp = pd.DataFrame(rows); cmp.to_csv(C.OUTPUTS / 'powell_vs_warsh_tests.csv', index=False); cmp.round(3)""")
code("""# The same comparison against Powell's last year only (Jun 2025 - May 2026), which is the fairer baseline
last = scores[(scores.chair == 'Powell') & (scores.date >= '2025-05-22')]
rows = []
for dt in ['statement', 'minutes', 'speech']:
    for s in C.TONE_SCORES:
        rows.append(dict(doc_type=dt, score=s, powell_last12m=last[last.doc_type == dt][s].mean(),
                         warsh=scores[(scores.doc_type == dt) & (scores.chair == 'Warsh')][s].mean()))
pd.DataFrame(rows).round(3)""")
code("""# Which themes drive the word-list score in statements, by Chair
theme_cols = [c for c in scores.columns if c.startswith(('hawk_', 'dove_'))]
scores[scores.subtype == 'statement'].groupby('chair')[theme_cols].mean().round(2).T""")

md("""## 3. Market reactions

Day 0 is the release day if the release is before 16:00 ET on a trading day, else the next trading day.
Changes are close-to-close: percent for DXY, basis points for 10s2s and the 1-year yield, IWF return minus IWN
return in percent for growth-minus-value. The control is the same-day change in the 3-month bill (DGS3MO), in bp.""")
code("""df = attach_market(scores, panel)
df = add_tone_changes(df)
print('releases with a usable day 0 and market data:', df[CHANGE_COLS + ['dDGS3MO']].dropna().shape[0], 'of', len(df))
df[CHANGE_COLS + ['dDGS3MO']].describe().round(2)""")
code("""# release-day moves vs all other days: is anything happening on Fed days at all?
from src.market import one_day_changes
allch = one_day_changes(panel).loc['2018-02-05':]
fed_days = set(df.day0.dropna())
print('abs one-day change, mean: Fed release days vs other days')
pd.DataFrame({'release days': allch[allch.index.isin(fed_days)].abs().mean(),
              'other days': allch[~allch.index.isin(fed_days)].abs().mean()}).round(2)""")

md("### Table 2. Warsh-era releases: tone scores next to the one-day change in the four indicators")
code("""t2 = X.table2(df)
t2""")

md("""### Table 3. Each indicator's one-day change regressed on each tone score, with the 3-month bill control

$\\Delta y_i = a + b\\,z(\\text{tone}_i) + c\\,\\Delta \\text{DGS3MO}_i + \\text{type dummies} + e_i$, HC1 standard errors.
Cells show $b$ (t-statistic); a one-unit move in the regressor is one standard deviation of the tone score.
Stars: * p<0.10, ** p<0.05, *** p<0.01.""")
code("""t3_level = run_table(df, '', type_dummies=True)
t3_level.to_csv(C.OUTPUTS / 'table3_level_pooled.csv', index=False)
print('Panel A. All releases (statements, minutes, speeches), tone level, N =', t3_level.n.min(), '–', t3_level.n.max())
format_table(t3_level, CHANGE_LABELS)""")
code("""t3_stmt = run_table(df[df.doc_type == 'statement'], '', type_dummies=False)
t3_stmt.to_csv(C.OUTPUTS / 'table3_level_statements.csv', index=False)
print('Panel B. Post-meeting statements only, tone level, N =', t3_stmt.n.min(), '–', t3_stmt.n.max())
format_table(t3_stmt, CHANGE_LABELS)""")
code("""t3_chg = run_table(df, '_chg', type_dummies=True)
t3_chg.to_csv(C.OUTPUTS / 'table3_change_pooled.csv', index=False)
print('Panel C. All releases, change in tone since the previous document of the same type (surprise proxy)')
format_table(t3_chg, CHANGE_LABELS)""")
code("""t3_stmt_chg = run_table(df[df.doc_type == 'statement'], '_chg', type_dummies=False)
t3_stmt_chg.to_csv(C.OUTPUTS / 'table3_change_statements.csv', index=False)
print('Panel D. Statements only, change in tone')
format_table(t3_stmt_chg, CHANGE_LABELS)""")
code("""# the control on its own: how much of each indicator's release-day move the rate decision explains
import statsmodels.api as sm
rows = []
for y in CHANGE_COLS:
    sub = df[[y, 'dDGS3MO']].dropna()
    m = sm.OLS(sub[y], sm.add_constant(sub['dDGS3MO'])).fit(cov_type='HC1')
    rows.append(dict(indicator=CHANGE_LABELS[y], b_bill=m.params['dDGS3MO'], t=m.tvalues['dDGS3MO'], r2=m.rsquared, n=int(m.nobs)))
pd.DataFrame(rows).round(3)""")
code("""# full detail behind Table 3, Panel A
t3_level.round(3)""")

md("""## 4. Inputs to the forecast

The forecast in the report combines (i) where Warsh's tone sits relative to Powell's, (ii) the estimated
response of each indicator to tone and to the bill change, and (iii) the Warsh-era release-day record.""")
code("""# 1. Warsh's statements relative to the Powell distribution
stm = scores[scores.subtype == 'statement'].sort_values('date')
for s in C.TONE_SCORES:
    p = stm[stm.chair == 'Powell'][s]
    print(f"{s:8s} Powell mean {p.mean():+.3f} sd {p.std():.3f} | Warsh Jun {stm.iloc[-2][s]:+.3f}  Jul {stm.iloc[-1][s]:+.3f}")
stm.tail(6)[['date', 'chair', 'n_words', 'hawk', 'dove'] + C.TONE_SCORES].round(3)""")
code("""# 2. Statement-to-statement tone: how often is a statement more hawkish than the previous one, and after a hawkish one?
for s in C.TONE_SCORES:
    d = stm[s].diff()
    up = (d > 0).mean()
    # conditional on the previous statement being in the top quartile of hawkishness
    hi = stm[s].shift(1) > stm[s].quantile(0.75)
    print(f"{s:8s} P(more hawkish than previous) = {up:.2f}; when previous was top-quartile hawkish = {(d[hi] > 0).mean():.2f}  (n={hi.sum()})")""")
code("""# 3. Release-day direction: how often does each indicator rise on a statement day, overall and on hawkish-statement days
st = df[df.doc_type == 'statement'].dropna(subset=CHANGE_COLS)
hawk = st.lex_net > st.lex_net.median()
pd.DataFrame({'P(rise) all statement days': (st[CHANGE_COLS] > 0).mean(),
              'P(rise) | hawkish statement': (st.loc[hawk, CHANGE_COLS] > 0).mean(),
              'P(rise) | bill up on the day': (st.loc[st.dDGS3MO > 0, CHANGE_COLS] > 0).mean(),
              'mean change | hawkish': st.loc[hawk, CHANGE_COLS].mean(),
              'mean |change| all': st[CHANGE_COLS].abs().mean()}).round(2)""")
code("""# 4. The market's own pricing going into 16 September: 3-month bill vs the middle of the target range
last = panel.dropna(subset=['DGS3MO']).iloc[-1]
print('as of', panel.dropna(subset=['DGS3MO']).index[-1].date(), 'DGS3MO =', last.DGS3MO, ' DGS1 =', last.DGS1, ' 10s2s =', last.T10Y2Y, ' DXY =', round(last.DXY, 2))
print('target range 3.50-3.75; a 3m bill at', last.DGS3MO, 'sits', round((last.DGS3MO - 3.625) * 100), 'bp above the midpoint')""")
code("""# 5. Warsh-era release days, all together
df[df.chair == 'Warsh'][['date', 'subtype'] + C.TONE_SCORES + CHANGE_COLS + ['dDGS3MO']].round(3)""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "analysis.ipynb")
print("wrote analysis.ipynb with", len(cells), "cells")
