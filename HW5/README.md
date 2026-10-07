# Assignment 5: What Voters Care About in the 2026 Midterms, How the Mood Around It Moves the Odds of a Democratic Sweep, and Which Stocks Care

FRE-GY 7871 A · NLP and the Investment Process · Fall 2026
Yuri Moghaddam Nasrollahi (ym3414)

Topic modelling (BERTopic) on ~80,000 news headlines and social-media posts about the midterms (Google News, Bluesky,
Mastodon, Reddit, Lemmy; January to October 2026) to find the issues voters care about; a validated three-model
sentiment module (RoBERTa tone, VADER, aspect-based partisan direction) aggregated into a weekly Key-Topic Sentiment
Index; correlation of that index with the Polymarket and Kalshi prices of a Democratic sweep of Congress; and a
pre-specified long-short "midterm basket" of 47 policy-exposed stocks whose market-adjusted performance is set against
both.

**The report is [`REPORT.md`](REPORT.md) (submitted on Brightspace as [`report.pdf`](report.pdf)). AI use is disclosed
in [`AI_USE.md`](AI_USE.md).** Raw corpora, model caches and embeddings are not committed (see `.gitignore`); the
`collect_*.py` scripts rebuild them. Every table and figure in the report is under `output/`.

## Reproduce

```bash
python3.12 -m venv .venv && source .venv/bin/activate      # or: uv venv --python 3.12 .venv
pip install -r requirements.txt
python collect_news.py discovery        # ~45 min: Google News RSS, 2 issue-free election queries x 278 days
python collect_bluesky.py               # ~1 h:    Bluesky top-100 'midterms' posts a day (the API firewall needs ~12 s between calls)
python collect_mastodon.py              # ~5 min:  mastodon.social midterm hashtag timelines
python collect_lemmy.py                 # ~10 min: lemmy.world comments and posts mentioning 'midterm'
python collect_reddit.py                # hours:   Arctic Shift full-text search is rate limited; windows are sampled in random order
python collect_markets.py && python collect_markets.py hourly   # Polymarket + Kalshi daily and 4pm-ET prices
python market_data.py                   # Yahoo Finance: 47 stocks, SPY, sector ETFs, Brent, 10y
python topic_model.py                   # BERTopic -> output/topic_cards.md (then label topics in data/topic_labels.json)
python collect_news.py topics           # stage 2: one issue-anchored Google News query per key issue per day
python sentiment.py                     # RoBERTa + VADER + ABSA on every document (cached)
python validate_sentiment.py            # 200 hand-labelled documents -> output/table_validation.csv
python sentiment_index.py               # issue salience, weekly issue sentiment, the KTSI
python equities.py                      # market model, midterm basket, election betas
python analysis.py                      # correlations, lead-lag, Granger, basket regressions, figures
python build_report.py                  # tables into REPORT.md, REPORT.html, report.pdf
```

## Pipeline

| Step | Script | Output |
|---|---|---|
| Sample, sources, paths | `config.py` | |
| News headlines (Google News RSS), stage 1 (issue-free) and stage 2 (issue-anchored); a publication-date guard drops items for which Google ignored the date operators | `collect_news.py`, `corpus.py` | `data/news_*.jsonl` |
| Social media: Bluesky, Mastodon, Reddit (Arctic Shift), Lemmy | `collect_bluesky.py`, `collect_mastodon.py`, `collect_reddit.py`, `collect_lemmy.py` | `data/*.jsonl`, `data/reddit/` |
| One document table (cleaning, ET dates, de-duplication, every social post must mention the midterms) | `corpus.py` | `data/corpus.parquet` |
| Topic discovery: mpnet embeddings, UMAP, HDBSCAN, c-TF-IDF; LLM-assisted labelling of clusters into voter issues | `topic_model.py`, `data/topic_labels.json` | `output/topic_info.csv`, `output/topic_cards.md` |
| Document sentiment: RoBERTa (tweets), VADER, aspect-based partisan direction (DeBERTa-v3 ABSA) | `sentiment.py` | `data/doc_sentiment.parquet` |
| Validation against 200 labelled documents | `validate_sentiment.py` | `output/table_validation.csv` |
| Issue salience, weekly issue sentiment, Key-Topic Sentiment Index | `sentiment_index.py` | `output/table_salience.csv`, `data/ktsi_weekly.csv` |
| Event contracts: Polymarket "Balance of Power: 2026 Midterms", Kalshi "Congress balance of power combo" | `collect_markets.py` | `data/pm_daily.csv`, `data/pm_4pm.csv` |
| Midterm basket: CAPM abnormal returns, channel legs, election betas, stability, event days | `market_data.py`, `equities.py` | `output/table_channels.csv`, `output/table_equities.csv` |
| Correlations and figures | `analysis.py`, `plotstyle.py` | `output/table_*.csv`, `output/fig/*.png` |
| Report | `build_report.py` | `REPORT.md`, `REPORT.html`, `report.pdf` |

## Conventions

- **Sample.** 1 January to 5 October 2026 (the analysis date is 6 October; election day is 3 November). Weeks end on
  Sunday; the weekly analysis uses the 40 weeks ending 4 January to 4 October.
- **P(Democratic sweep).** Probability that Democrats control both the House and the Senate after the election: the
  YES price of Polymarket's "D Senate, D House" outcome and Kalshi's "Democrats sweep" outcome (bid-ask midpoint),
  averaged. Daily values are the last print of each US/Eastern day; for stock regressions, the print at 16:00 ET.
- **Key-Topic Sentiment Index (KTSI).** Salience-weighted average over the seven key issues of the weekly mean
  document score, after removing each source's baseline for that issue and weighting sources equally. Two versions:
  tone (RoBERTa P(pos) − P(neg)) and pro-Democratic direction (aspect-based).
- **Market-adjusted performance.** Market-model (CAPM) abnormal returns with alpha and beta estimated on 2025 daily
  returns against SPY; the basket is long the Democratic-sweep-winner channels and short the Republican-hold-winner
  channels, equal weight within and across channels.
