# AI use disclosure

**Tools used:**

Claude (Opus 5). I would put its share of the work at roughly 15 percent. It was a lookup and
debugging tool and did not touch the design of the study or the reading of the results.

**What I used it for:**

- **Setup and boilerplate.** The `requirements.txt` pins, the `.gitignore` rule that keeps `data/`
  out of the repository, and the on-disk caching wrapper in `src/collect.py` that makes the
  downloads resumable. Roughly forty lines, none of which affects a number in the report.
- **Debugging, three times.** (i) The Board's speech feed has one entry dated `11/28/2006` with no
  time of day, and `datetime.strptime` with a single format string died on it halfway through a
  ten-minute download; the model suggested the fallback format list now in `_parse_dt`. (ii)
  `yfinance` returned a MultiIndex on the columns, so `px['Close']` raised, and I used it to confirm
  that `get_level_values(0)` was the right flattening rather than `droplevel`. (iii) A matplotlib
  date axis would not take a `YearLocator` until the series index was a real `DatetimeIndex`. In all
  three I had already isolated the failing line.
- **API lookups.** The `cov_type='HC1'` spelling for `statsmodels` `.fit()`, and how `transformers`
  exposes the last hidden layer when `output_hidden_states=True` so I could mean-pool it for the
  anchor-similarity score.
- **One proofreading pass on the report**, for typos, unclear sentences, and any number in the prose
  that disagreed with the tables.

**What I wrote myself:**

The design and everything that follows from it. The sample definition and the Chair-term split; the
decision to treat press-conference transcripts as the Chair's speech and to keep only the Chair's
own words; the 111-phrase monetary-policy word list and its four themes, which I built by reading
statements from 2018, 2019, 2021 and 2022 and writing down what actually distinguishes them; the
negation rule; the eight paired anchor sentences for the FinBERT similarity score; the day-0 rule
and the close-to-close change conventions; the choice to report tone levels and tone changes as
separate panels; the 3-month bill control; the HC1 standard errors; every exhibit; the reading of
the results; and the forecast and the recommended position. The notebook is mine cell by cell.

I kept the model away from the parts of this assignment where its output is hardest to check:
choosing the specification, deciding which result is the finding, and writing the interpretation.

**Anything the model got wrong that I had to correct:**

1. **A suggested fix that treated the symptom.** When the minutes count came back at 45 instead of
   the 66 I expected, the suggestion was to loosen the regular expression that matches the release
   date. That would have matched the wrong date on the historical calendar pages, where a second
   "Released" string appears further down. The real problem was the 600-character search window,
   which is wide enough on the current calendar page and too narrow on the pre-2021 ones. I widened
   the window and checked the parsed release dates against the calendar by hand for 2018.
2. **A function signature from a newer version.** A suggested `pandas` `pct_change` call passed
   `fill_method=None` in a position that the pinned version does not accept, and raised immediately.
3. **A wording change in the proofreading pass that altered a claim.** My sentence about the Panel A
   coefficients — "nothing within 1.6 standard errors of zero" — came back as "tone has no effect on
   asset prices", which is a much stronger statement than a null result on 307 close-to-close
   observations supports. I reverted it.

The two errors I found myself were both in the corpus rather than in the code, and neither announced
itself. Four press releases matching the pattern `monetaryYYYYMMDDa.htm` are not FOMC statements at
all — the Statement on Longer-Run Goals in August 2020 and August 2025, the October 2019
implementation note, and a March 2020 facility announcement — and they sat in the sample scoring as
statements until I read the titles. And every page I had cached was mojibake, because
`federalreserve.gov` does not declare a charset in the header and `requests` fell back to
ISO-8859-1; the word list did not care, but FinBERT sees the punctuation, so I set the encoding and
re-scored the corpus from scratch.

That is the pattern I would draw out of this. The model's mistakes were in code, surfaced within
seconds, and cost minutes. The mistakes that reached the scored corpus were mine, were silent, and
would have changed Table 1 and every FinBERT column if I had not gone looking.
