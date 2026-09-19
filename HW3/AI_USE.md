# AI use disclosure

**Tools used:**

Claude (Fable 5.1, through Claude Code). Its share of the work on this assignment was large: it wrote the code,
ran the pipeline, and drafted the report, working from my instructions and the two papers I gave it. I set the
question (apply the Rigobon and Sack method to the 2026 Iran war, with the war-news days chosen by NLP rather than
by hand, on global as well as US variables), chose the sample, reviewed each stage, and checked the numbers and the
interpretation before submitting.

**What I used it for:**

- **Reading the two papers** and turning equations 1 to 12 of Rigobon and Sack (2003) into `warrisk.py`: the
  three IV estimators, robust standard errors, the nearest-day rule for the low-variance days, and the Table 3
  variance decomposition.
- **Data collection.** FRED and Yahoo Finance downloads (`market_data.py`); Google News RSS headlines, Wikipedia
  current-events pages and the Wikipedia war timeline (`collect_news.py`, `wiki_timeline.py`). GDELT was tried
  and rate-limited; the code keeps the hook.
- **The NLP index** (`nlp_pipeline.py`): the relevance regexes, the escalation/de-escalation lexicon, FinBERT scoring,
  the TF-IDF novelty measure, and the composite index and its mapping to trading days.
- **The seven specifications** in `run_analysis.py`, the hand-curated event list used as a comparison
  (`data/curated_events.csv`, assembled from the Wikipedia and CNBC/PBS timelines), the figures, and the
  appendix tables.
- **Drafting the report**, including the comparison with the 2003 results and the limitations section.
- **Environment fixes.** After a macOS upgrade, scipy's Fortran extensions stopped loading and broke `transformers`
  through `sklearn`; `compat.py` stubs those modules so FinBERT runs. This does not affect any number.

**What I did myself:**

Chose the topic, the sample window and the set of variables; decided to keep the paper's two-year-yield
normalisation for the pre-war window and to switch to the Brent normalisation for the full sample once the
variance check showed the two-year yield does not qualify in 2026; reviewed the selected war-news days in Table 1
against what actually happened on those dates; and read every section of the report before submitting it.

**Things to be aware of when reading the results:**

1. The news-to-trading-day mapping is by calendar date, so events after the US close (the first strikes on the
   evening of 27 February) are attributed to that day rather than the next trading day. Rigobon (2003, section IV)
   shows this lowers the variance contrast but does not bias the estimator.
2. The RSS feed caps headline volume at 100 per query per day, which mutes the volume component of the index
   during the sustained-combat weeks of March; specification S7 shows the effect.
3. The escalation lexicon was written by the model and is tilted toward escalation vocabulary; it is used only to
   label the direction of the selected days, not inside the estimator.
