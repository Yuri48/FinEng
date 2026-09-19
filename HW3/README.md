# Assignment 3: The Effects of Iran War Risk on Global Financial Markets in 2026

FRE-GY 7871 A · NLP and the Investment Process · Fall 2026
Yuri Moghaddam Nasrollahi (ym3414)

Replication of Rigobon and Sack (2003), *The Effects of War Risk on U.S. Financial Markets* (NBER WP 9609), for the
2026 Iran war, with the "war news days" chosen by an NLP pipeline (headline volume, novelty, escalation lexicon,
FinBERT tone) instead of by hand, applied to 29 US and global financial variables.

**The report is [`REPORT.md`](REPORT.md) (submitted on Brightspace as [`report.pdf`](report.pdf)). AI use is disclosed in
[`AI_USE.md`](AI_USE.md).** The raw scraped corpora are not committed; the derived daily files under `data/` and every
table and figure under `output/` are.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python market_data.py        # ~1 min: FRED + Yahoo Finance -> data/levels.csv, data/changes_2026.csv
python collect_news.py       # ~15 min: Google News RSS (2 queries x 261 days) + Wikipedia current events (+ GDELT if not rate-limited)
python wiki_timeline.py      # ~1 min: Wikipedia war timeline and prelude pages -> data/wiki_timeline_daily.csv
python nlp_pipeline.py       # ~5 min: relevance filter, lexicon, FinBERT (19k headlines, cached), novelty -> data/daily_news_features.csv
python run_analysis.py       # ~1 min: index, H/L days, IV estimates for 7 specifications, figures -> output/
python annotate_days.py S1_level_q75   # Table 1 analogue for a specification
python make_report.py && python build_html.py   # appendix tables -> REPORT.md tail, REPORT.html
```

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Market data (FRED + Yahoo Finance), daily changes in the paper's units | `market_data.py` | `data/levels.csv`, `data/changes_2026.csv`, `data/var_spec.json` |
| 2. News corpora: Google News RSS headlines (2 queries/day), Wikipedia *Portal:Current events* daily pages, GDELT timelines (if API allows) | `collect_news.py` | `data/gnews_headlines.jsonl`, `data/wiki_current_events.jsonl`, `data/gdelt_*.csv` |
| 3. Wikipedia *Timeline of the 2026 Iran war* + prelude pages parsed into daily text | `wiki_timeline.py` | `data/wiki_timeline_daily.csv` |
| 4. NLP features per calendar day: relevance filter, escalation lexicon, FinBERT tone, TF-IDF novelty, volume | `nlp_pipeline.py` | `data/daily_news_features.csv`, `data/relevant_headlines.csv` |
| 5. Composite war-news index, H/L day selection, IV estimates (three instruments), variance decomposition, six specifications, figures | `run_analysis.py` (uses `warrisk.py`) | `output/table2_*.csv`, `output/summary_across_specs_*.csv`, `output/H_days_*.csv`, `output/fig/*.png` |
| 6. Table-1 analogue: headlines and market moves on each selected H day | `annotate_days.py <spec>` | `output/table1_<spec>.csv` |

Run in order with `python3 market_data.py && python3 collect_news.py && python3 wiki_timeline.py && python3 nlp_pipeline.py && python3 run_analysis.py && python3 annotate_days.py S1_level_q75`.
`compat.py` stubs a few scipy/sklearn modules whose compiled extensions fail to load on this machine after a macOS upgrade; nothing in the project uses them.

## Method (Rigobon & Sack 2003, eq. 1-12)

Daily changes of financial variables `x` are driven by common factors `z` (war risk `z1` plus everything else) and idiosyncratic
shocks. The impact of `z1` on the two-year Treasury yield is normalised to 1 and its impact `d21` on any other variable is
identified from the *shift* of the covariance matrix between "war news" days (H) and nearby other days (L), assuming only the
variance of `z1` changes: `dOmega = dVar(z1) * [1, d21; d21, d21^2]`. The three IV estimators are

* `nu1 = +dx1 on H, -dx1 on L`  ->  `d21 = dCov / dVar(x1)`
* `nu2 = +dx2 on H, -dx2 on L`  ->  `d21 = dVar(x2) / dCov`
* `nu3 = [nu1, nu2]`             ->  2SLS with both (the paper's preferred column)

Coefficients are reported for a war-risk increase that lowers the two-year yield by 25 bp (`scale = -0.25`), and for
robustness for one that raises the Brent front future by $5. Table 3 of the paper (variance on L and H days, predicted
change in variance `d^2 * dVar(x1)`, share explained on H days and over the whole window) is reproduced for each variable.

## NLP index

For each calendar day: `n_rel` conflict-relevant Iran headlines; `share_rel` share of Iran coverage that is conflict
related; Wikipedia current-events and war-timeline text volumes; `novelty` = 1 - cosine similarity of the day's TF-IDF
headline vector to the previous five days (new information); `esc_abs_chg` = absolute day-to-day swing in an escalation
lexicon score (direction surprises); `tone_disp` = cross-headline dispersion of FinBERT negativity (disagreement). The
z-scored components are averaged, calendar days are mapped to the next US trading day, and H days are the top quantile.
