# AI use disclosure:

**Tools used:**

Claude (through Claude Code). I would put its share of the work at roughly 20 percent. It was a coding assistant for the
data-collection plumbing and a debugging and lookup tool; it did not choose the question, the sources, the issues, the
stocks or the interpretation.

**What I used it for:**

- **Collector boilerplate and source work-arounds.** The resumable request loops in the `collect_*.py` scripts and the
  retry logic. Three source problems where I had found the symptom and asked for the fix: Bluesky's AppView returning 403
  to Python's HTTP client but not to curl (it suggested routing the calls through the system curl, now in
  `collect_bluesky.py`), the Arctic Shift archive answering full-text searches with a 422 "slow down" (it pointed me to the
  `x-ratelimit-reset` header for the back-off), and Kalshi daily candles that gave two rows for one Eastern-time day
  in the weeks around the daylight-saving change.
- **Debugging, four times.** (i) `transformers` failed to load the DeBERTa tokenizer until `sentencepiece` and
  `tiktoken` were installed. (ii) `grangercausalitytests` in the installed `statsmodels` no longer accepts `verbose`.
  (iii) The aspect-based scoring run grew to 5 GB and started swapping because PyTorch's MPS allocator keeps buffers per
  padded batch shape; it suggested sorting by length, smaller batches and `torch.mps.empty_cache()`. (iv) Headless Chrome
  hung on PDF printing behind my running Chrome; the separate profile and keychain flags in `build_report.py` came from
  that exchange. In each case I had already isolated the failing call.
- **API lookups.** The Polymarket `prices-history` limits (14-day windows for hourly data), the Kalshi candlestick
  endpoint, the BERTopic `reduce_outliers` signature, and the `cov_type="HAC"` options in `statsmodels`.
- **Presentation code.** The Markdown-to-HTML converter in `build_report.py` and the shared matplotlib style in
  `plotstyle.py`. Neither affects a number.
- **One proofreading pass on the report**, for typos, unclear sentences, and numbers in the text that disagreed with the
  tables.

**What I did myself:**

The design and everything that follows from it. The two-stage news design (issue-free discovery queries, then
issue-anchored queries) and the choice of platforms and Reddit communities; the rule that every social post must mention
the midterms; the BERTopic settings and the outlier rule; **reading all 89 topic cards and labelling each cluster and its
issue** (Appendix A); the choice of the eight key issues and the comparison with the September polls; the three sentiment
measures and the decision to score partisan direction with an aspect model rather than flip document tone; **hand-labelling
the 200 validation documents**; the source-demeaned, salience-weighted index; the weekly design and the decision to rest
inference on changes rather than levels; the twelve policy channels, the 47 stocks and their expected signs; the 2025
estimation window and the 4 p.m. alignment of the event contracts; every exhibit; the reading of the results; and the
report, which I wrote section by section.

I kept the tool away from the steps whose output is hardest to check: the topic labels, the validation labels, the stock
selection, and the interpretation.

**Anything the tool got wrong that I had to correct:**

1. **A rate-limit fix that would have made things worse.** For Bluesky's 403s, the first suggestion was to run several
   workers in parallel. The 403s came from the edge firewall reacting to bursts, so more workers meant a longer block.
   Slowing to one call every 12 seconds, with two workers walking the calendar from opposite ends, is what worked.
2. **A plotting suggestion that broke a rule I had set.** It proposed a dual-axis chart of the sweep probability against
   the sentiment index; I used stacked panels instead, because the alignment of two y-scales is arbitrary and invents
   co-movement.
3. **A wording change in the proofreading pass that overstated a result.** My sentence that the composite index "carries
   almost no information about the weekly changes in the odds" came back as "sentiment does not affect election odds",
   which is a causal claim the correlations cannot support. I reverted it.

The errors that reached the data were mine, and none of them announced itself. One version of the stage-2 democracy query
was long enough that Google silently ignored its date operators and returned the same day's headlines for every date; I
found it because 23,500 headlines had 204 distinct titles, re-ran the query in a shorter form, and added the
publication-date guard in `corpus.py`. The corpus loader read only one of the two Bluesky files, which dropped 6,960 posts
from 16 July to 5 October, the period when the odds moved most; a gap in the weekly document counts in Figure 2 gave it
away. And the same date guard, applied to the weekly-window stock-news queries, threw away five-sixths of those headlines
until I made it window-aware. As in earlier assignments, the tool's mistakes were in code and surfaced in seconds; mine
were in the data and would have changed Table 2, the late-sample index and the news screen if I had not gone looking.
