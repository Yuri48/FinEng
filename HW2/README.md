# Assignment 2: Evaluating the Impact of FOMC Communications on Asset Prices

FRE-GY 7871 A · NLP and the Investment Process · Fall 2026
Yuri Moghaddam Nasrollahi (ym3414)

Kevin Warsh took the oath as Chair of the Federal Reserve on 22 May 2026. This repository
measures how the tone of Fed communication changed when he replaced Jerome Powell, tests
whether that tone moved four market indicators on release days after controlling for the
rate decision, and turns the result into a forecast for the 15–16 September 2026 FOMC meeting.

**The notebook with saved output is [`analysis.ipynb`](analysis.ipynb). The report is
`REPORT.md` (submitted on Brightspace as a PDF). AI use is disclosed in [`AI_USE.md`](AI_USE.md).**
No data files are committed; everything under `data/` is rebuilt by the three scripts below.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/01_collect_documents.py   # ~10 min: 308 documents from federalreserve.gov, cached in data/raw
python scripts/02_score_tone.py          # ~25 min on an M-series Mac (FinBERT on ~60k sentences), cached per document
python scripts/03_get_market_data.py     # ~1 min: FRED + Yahoo Finance
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```

## What is where

| Path | What it does |
|---|---|
| `src/config.py` | Sample definition: Chair start dates, indicators, release-time defaults, paths. |
| `src/collect.py` | Scrapes statements and minutes (from the FOMC calendar pages), the Chair's speeches and testimony (from the Board's JSON feeds), and press-conference transcripts (PDF; keeps only the Chair's own words). Records each document's release date and time. |
| `src/lexicon.py` | The monetary-policy word list: 56 hawkish and 55 dovish phrase patterns in four themes (inflation, activity, labor, policy), with negation handling. Score = (H − D)/(H + D). |
| `src/finbert.py` | FinBERT (`ProsusAI/finbert`) sentence sentiment (P(pos) − P(neg)) and hawkish-minus-dovish anchor-sentence similarity, from one forward pass per sentence. |
| `src/market.py` | DXY, 10s2s, 1-year yield, IWF − IWN, and the DGS3MO control; the day-0 rule and one-day changes. |
| `src/regress.py` | One-day change on standardised tone with the 3-month-bill control, HC1 standard errors; level and change-in-tone versions. |
| `src/exhibits.py` | Table 1, Figure 1, Table 2 and the by-Chair summary. |
| `scripts/` | The three pipeline steps. |
| `outputs/` | The tables and figure used in the report (small CSV/PNG files, committed). |

## Conventions that need exact definitions

- **Sample.** Powell's term from 5 February 2018 to 21 May 2026 is the baseline; Warsh from 22 May 2026 to 14 September 2026.
- **Documents.** *Statements*: every post-meeting statement, including the two unscheduled March 2020 statements. *Minutes*: dated by their release day, three weeks after the meeting. *Speeches*: the Chair's speeches and congressional testimony from the Board feeds plus the Chair's portion of every press-conference transcript.
- **Day 0.** The release date when the release is before 16:00 ET on a trading day; otherwise the next trading day. Changes are close-to-close: percent for DXY, basis points for the two yield series, and IWF daily return minus IWN daily return in percent for growth-minus-value.
- **Tone scores** are standardised within the regression sample, so a coefficient is the effect of a one-standard-deviation more hawkish document.
