"""Market indicators: FRED (T10Y2Y, DGS1, DGS3MO) and Yahoo Finance (DX-Y.NYB, IWF, IWN).

One-day change convention (same day-0 rule as in Assignment 1):
  day 0 = the release date if it is a trading day and the release is before 16:00 ET,
          otherwise the next trading day.
  change = level on day 0 minus level on the previous trading day
           (percent change for DXY, basis points for yields and the spread,
            IWF return minus IWN return in percent for growth-minus-value).
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

from . import config as C

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def fetch_fred(series: str) -> pd.Series:
    path = C.MARKET / f"{series}.csv"
    if not path.exists():
        r = requests.get(FRED_CSV.format(series), timeout=60)
        r.raise_for_status()
        path.write_text(r.text)
    df = pd.read_csv(path, na_values=".")
    df.columns = ["date", series]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[series].astype(float)


def fetch_yahoo(ticker: str) -> pd.Series:
    import yfinance as yf
    path = C.MARKET / f"{ticker.replace('^', '')}.csv"
    if not path.exists():
        px = yf.download(ticker, start="2017-12-01", auto_adjust=True, progress=False)
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = px.columns.get_level_values(0)
        px[["Close"]].to_csv(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["Close"].astype(float).rename(ticker)


def build_panel() -> pd.DataFrame:
    """Daily levels of every indicator on the equity trading calendar (IWF)."""
    iwf, iwn, dxy = fetch_yahoo("IWF"), fetch_yahoo("IWN"), fetch_yahoo("DX-Y.NYB")
    cal = iwf.index
    panel = pd.DataFrame(index=cal)
    panel["DXY"] = dxy.reindex(cal)
    panel["IWF"] = iwf
    panel["IWN"] = iwn.reindex(cal)
    for s in ("T10Y2Y", "DGS1", "DGS3MO"):
        panel[s] = fetch_fred(s).reindex(cal)
    panel.to_csv(C.MARKET / "panel.csv")
    return panel


def load_panel() -> pd.DataFrame:
    p = C.MARKET / "panel.csv"
    return pd.read_csv(p, index_col=0, parse_dates=True) if p.exists() else build_panel()


def day_zero(dates: pd.Series, times: pd.Series, cal: pd.DatetimeIndex) -> pd.Series:
    """Map (release date, HH:MM ET) to the trading day whose close first reflects the release."""
    out = []
    for d, t in zip(pd.to_datetime(dates), times):
        d = pd.Timestamp(d).normalize()
        hh = int(str(t)[:2]) if pd.notna(t) else 12
        if hh >= 16:
            d = d + pd.Timedelta(days=1)
        i = cal.searchsorted(d)
        out.append(cal[i] if i < len(cal) else pd.NaT)
    return pd.Series(out, index=dates.index)


def one_day_changes(panel: pd.DataFrame) -> pd.DataFrame:
    """Per trading day: the one-day change of each indicator and of the control."""
    ch = pd.DataFrame(index=panel.index)
    ch["dDXY"] = panel["DXY"].pct_change(fill_method=None) * 100
    ch["d10s2s"] = panel["T10Y2Y"].diff() * 100
    ch["dDGS1"] = panel["DGS1"].diff() * 100
    ch["dGmV"] = (panel["IWF"].pct_change(fill_method=None) - panel["IWN"].pct_change(fill_method=None)) * 100
    ch["dDGS3MO"] = panel["DGS3MO"].diff() * 100
    return ch


CHANGE_COLS = ["dDXY", "d10s2s", "dDGS1", "dGmV"]
CHANGE_LABELS = {"dDXY": "DXY (%)", "d10s2s": "10s2s (bp)", "dDGS1": "1y yield (bp)",
                 "dGmV": "Growth − value (%)", "dDGS3MO": "3m bill (bp)"}


def attach_market(corpus: pd.DataFrame, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    panel = panel if panel is not None else load_panel()
    ch = one_day_changes(panel)
    df = corpus.copy()
    df["day0"] = day_zero(df["date"], df["time"], panel.index)
    df = df.join(ch, on="day0")
    return df
