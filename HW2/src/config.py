"""Sample definition, paths and constants shared by every script and the notebook."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"            # cached HTML / PDF from federalreserve.gov (not committed)
INTERIM = DATA / "interim"    # parsed corpus and tone scores (not committed)
MARKET = DATA / "market"      # FRED and Yahoo downloads (not committed)
OUTPUTS = ROOT / "outputs"    # tables and figures that go into the report (committed)

for p in (RAW, INTERIM, MARKET, OUTPUTS):
    p.mkdir(parents=True, exist_ok=True)

# Chair terms. Powell took the oath on 5 Feb 2018; Warsh on 22 May 2026.
POWELL_START = "2018-02-05"
WARSH_START = "2026-05-22"
SAMPLE_END = "2026-09-14"

FED = "https://www.federalreserve.gov"
USER_AGENT = "FRE-GY-7871A student project (NYU) contact via course"

# Release times, US/Eastern, when the Fed page does not carry one.
STATEMENT_TIME = "14:00"
MINUTES_TIME = "14:00"
PRESSCONF_TIME = "14:30"

DOC_TYPES = ["statement", "minutes", "speech"]  # speech = speeches + testimony + press conferences

# Market indicators: (name, source, ticker, how the one-day change is measured)
INDICATORS = {
    "DXY":   dict(source="yahoo", ticker="DX-Y.NYB", change="pct", unit="%"),
    "10s2s": dict(source="fred",  ticker="T10Y2Y",   change="diff_bp", unit="bp"),
    "DGS1":  dict(source="fred",  ticker="DGS1",     change="diff_bp", unit="bp"),
    "GmV":   dict(source="yahoo", ticker=("IWF", "IWN"), change="ret_diff", unit="%"),
}
CONTROL = dict(source="fred", ticker="DGS3MO", change="diff_bp", unit="bp")

TONE_SCORES = ["lex_net", "fb_sent", "fb_sim"]
TONE_LABELS = {
    "lex_net": "Word list (net hawkish)",
    "fb_sent": "FinBERT sentiment",
    "fb_sim": "FinBERT similarity",
}
