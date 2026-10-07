"""Sample definition, sources and paths shared by every step of the pipeline."""
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA, OUT, FIG = ROOT / "data", ROOT / "output", ROOT / "output" / "fig"
for p in (DATA, OUT, FIG):
    p.mkdir(parents=True, exist_ok=True)

# Sample: first trading day of 2026 to the last complete day before the analysis date (6 Oct 2026).
# Election day is Tuesday 3 November 2026.
START, END = date(2026, 1, 1), date(2026, 10, 5)
ELECTION_DAY = date(2026, 11, 3)
UA = {"User-Agent": "Mozilla/5.0 (NYU FRE-GY 7871 HW5 student project)"}

# ---- News: Google News RSS (100 items per query per day) -------------------------------------------------------
# Stage 1 (topic discovery) uses two broad election queries that do not name any issue, so the topic model, not
# the query, decides what the election is about.
NEWS_DISCOVERY_QUERIES = {
    "midterms": 'midterms OR "midterm elections" OR "midterm election"',
    "voters": 'voters (congress OR midterms OR "2026 election" OR "house race" OR "senate race")',
}

# ---- Social: Reddit through the Arctic Shift archive, comments that mention the midterms -----------------------
# Left, centre and right communities. The archive's full-text search is rate limited to a few calls a minute, so
# the design samples rather than enumerates: value = (window length in hours, window start hour UTC, step in hours).
# r/politics: one 6-hour US-evening window (00-06 UTC = 8pm-2am ET) every day, 100 comments newest first.
# Mid-sized subreddits: one 84-hour window twice a week. Small subreddits: one window a week.
SUBREDDITS = {
    "politics": (6, 0, 24), "PoliticalDiscussion": (84, 0, 84), "moderatepolitics": (84, 0, 84),
    "Conservative": (84, 0, 84), "democrats": (168, 0, 168), "Republican": (168, 0, 168),
    "AskConservatives": (168, 0, 168), "centrist": (168, 0, 168),
}
SUB_LEAN = {"politics": "left", "democrats": "left", "PoliticalDiscussion": "centre", "moderatepolitics": "centre",
            "centrist": "centre", "Conservative": "right", "Republican": "right", "AskConservatives": "right"}

# ---- Social: Mastodon hashtag timelines (mastodon.social federated view) ---------------------------------------
MASTODON_TAGS = ["midterms", "midterms2026", "midterm", "election2026", "2026midterms"]

# ---- Prediction markets: P(Democratic sweep) = Democrats win both House and Senate -----------------------------
POLYMARKET_EVENT = "balance-of-power-2026-midterms"
KALSHI_SERIES, KALSHI_EVENT = "KXBALANCEPOWERCOMBO", "KXBALANCEPOWERCOMBO-27FEB"
KALSHI_CHAMBERS = {"house_D": ("CONTROLH", "CONTROLH-2026-D"), "senate_D": ("CONTROLS", "CONTROLS-2026-D")}
