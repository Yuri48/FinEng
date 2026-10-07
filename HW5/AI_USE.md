# AI use disclosure

**Tools used:**

Claude (Opus 5.5, through Claude Code). Its share of the work on this assignment was large: from the assignment text,
it designed the study, wrote all of the code, collected the data, ran the models and drafted the report. Two steps use
the model *inside* the method rather than only as a coding assistant, and they are disclosed separately below because
they affect numbers in the report.

**What I used it for:**

- **Data collection.** Google News RSS headlines (`collect_news.py`), Bluesky (`collect_bluesky.py`), Mastodon
  (`collect_mastodon.py`), Reddit through the Arctic Shift archive (`collect_reddit.py`), Lemmy (`collect_lemmy.py`),
  Polymarket and Kalshi price histories (`collect_markets.py`) and Yahoo Finance prices (`market_data.py`), including
  the work-arounds each source needed (rate limits on Arctic Shift and Bluesky, the Bluesky edge firewall, and a
  publication-date guard after one Google News query silently ignored its date operators).
- **Topic modelling** (`topic_model.py`): BERTopic with mpnet embeddings, UMAP, HDBSCAN and c-TF-IDF.
- **Sentiment** (`sentiment.py`, `sentiment_index.py`): the choice of a tweet-trained RoBERTa for tone, VADER as the
  lexicon baseline, and an aspect-based model for partisan direction; the source-demeaned, salience-weighted weekly
  index.
- **The midterm basket** (`market_data.py`, `equities.py`): the policy channels, the 47 stocks, their expected signs,
  the market-model abnormal returns and the election-beta tests.
- **The correlation analysis, figures and report draft** (`analysis.py`, `build_report.py`, `REPORT.md`).

**Where the model is part of the method (affects reported numbers):**

1. **Topic labels.** HDBSCAN produces unlabelled clusters. The model read each cluster's c-TF-IDF keywords and six
   of its documents (`output/topic_cards.md`) and assigned a label and a voter issue in `data/topic_labels.json`.
   This is the "LLM as topic labeller" step BERTopic supports. The salience ranking in Table 2 depends on these
   assignments; the mapping is printed in the appendix so it can be checked cluster by cluster.
2. **Validation labels.** The 200 documents used to validate the sentiment models (Table 3) were labelled for tone and
   partisan direction by the model reading each text (`data/validation_labels.json`), not by a human annotator.
   Agreement figures should be read as agreement with an LLM annotator.

**What I did myself:**

*[To complete before submitting: e.g. which topic labels and validation labels you checked by hand, which parts of
the report you rewrote, and any decisions you changed.]*

**Things to be aware of when reading the results:**

1. The social-media sample leans left. Bluesky, Mastodon and Lemmy are left-of-centre platforms, and right-leaning
   voices enter mainly through three of the eight Reddit communities. Salience is reported by source for this
   reason, and the sentiment index removes each source's baseline before averaging; its *level* still should not be
   read as a measure of national opinion.
2. Reddit coverage is partial (a random subset of the planned windows) because the archive's full-text search was
   rate limited to roughly one call a minute.
3. Ten of the twelve basket channels were specified before any return data were examined. Two (AI power producers and
   big pharma) were added after a first estimate of the ten-channel basket had been seen, on the strength of the news
   screen; the ten-channel basket gives the same null result, which the report states. No channel was dropped.
