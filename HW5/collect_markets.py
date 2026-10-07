"""Daily price histories of the 2026 balance-of-power event contracts.

Polymarket: 'Balance of Power: 2026 Midterms' (five mutually exclusive outcomes), CLOB prices-history of each YES token.
Kalshi:     'Congress balance of power combo' (Democrats sweep / Republicans sweep / split outcomes) plus the House and
            Senate control markets, daily candlesticks (close, bid/ask, volume).
Every price is mapped to the US/Eastern calendar day it closes. -> data/pm_daily.csv
Hourly histories of the two Democratic-sweep contracts give the price at 16:00 ET on each trading day, so daily
changes line up with stock closes. -> data/pm_4pm.csv
"""
import json, time
from datetime import datetime, timezone
import pandas as pd
import requests
from config import DATA, UA, POLYMARKET_EVENT, KALSHI_SERIES, KALSHI_EVENT, KALSHI_CHAMBERS

ET = "America/New_York"
KAPI = "https://api.elections.kalshi.com/trade-api/v2"


def get(url, **params):
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=60)
            if r.status_code == 200:
                return r.json()
            print(url, r.status_code, flush=True)
        except Exception as e:
            print(url, e, flush=True)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(url)


def polymarket():
    ev = get("https://gamma-api.polymarket.com/events", slug=POLYMARKET_EVENT)[0]
    cols = {}
    meta = []
    for m in ev["markets"]:
        name = m["question"].replace("2026 Balance of Power: ", "")
        key = {"D Senate, D House": "poly_DD", "R Senate, R House": "poly_RR", "R Senate, D House": "poly_DhouseRsen",
               "D Senate, R House": "poly_RhouseDsen", "Other": "poly_other"}[name]
        yes = json.loads(m["clobTokenIds"])[0]
        h = get("https://clob.polymarket.com/prices-history", market=yes, interval="max", fidelity=1440)["history"]
        s = pd.Series({pd.Timestamp(x["t"], unit="s", tz="UTC").tz_convert(ET): x["p"] for x in h}).sort_index()
        s.index = s.index.date
        cols[key] = s.groupby(level=0).last()                     # last print of each ET day
        meta.append({"market": key, "question": m["question"], "token_yes": yes, "volume_usd": float(m.get("volume") or 0)})
        time.sleep(0.5)
    return pd.DataFrame(cols), meta


def kalshi_candles(series, ticker, start_ts):
    end_ts = int(datetime.now(timezone.utc).timestamp())
    c = get(f"{KAPI}/series/{series}/markets/{ticker}/candlesticks", start_ts=start_ts, end_ts=end_ts,
            period_interval=1440)["candlesticks"]
    rows = []
    for x in c:
        day = (pd.Timestamp(x["end_period_ts"] - 1, unit="s", tz="UTC").tz_convert(ET)).date()   # candle closes at ET midnight
        f = lambda d, k: float(d[k]) if d.get(k) not in (None, "") else None
        bid, ask = f(x["yes_bid"], "close_dollars"), f(x["yes_ask"], "close_dollars")
        rows.append({"date": day, "close": f(x["price"], "close_dollars"), "mid": (bid + ask) / 2 if bid is not None and ask is not None else None,
                     "volume": float(x.get("volume_fp") or 0)})
    return pd.DataFrame(rows).groupby("date").last()             # DST weeks can give two candles one ET day


def kalshi():
    ev = get(f"{KAPI}/events/{KALSHI_EVENT}", with_nested_markets="true")
    markets = ev.get("markets") or ev["event"].get("markets")
    out, meta = {}, []
    for m in markets:
        code = m["ticker"].split("-")[-1]                         # DD, RR, DR, RD
        start = int(pd.Timestamp(m["open_time"]).timestamp())
        k = kalshi_candles(KALSHI_SERIES, m["ticker"], start)
        out[f"kalshi_{code}"] = k["mid"].fillna(k["close"]); out[f"kalshi_{code}_vol"] = k["volume"]
        meta.append({"market": f"kalshi_{code}", "question": m.get("yes_sub_title"), "ticker": m["ticker"]})
        time.sleep(0.5)
    for key, (series, ticker) in KALSHI_CHAMBERS.items():
        k = kalshi_candles(series, ticker, int(pd.Timestamp("2025-06-01", tz="UTC").timestamp()))
        out[f"kalshi_{key}"] = k["mid"].fillna(k["close"])
        meta.append({"market": f"kalshi_{key}", "question": key, "ticker": ticker})
    return pd.DataFrame(out), meta


def hourly_4pm(start="2025-12-12", end=None):
    """Last hourly print at or before 16:00 ET on each weekday, for Polymarket D/D and Kalshi Democrats-sweep."""
    end = pd.Timestamp(end or pd.Timestamp.now(tz="UTC")).tz_convert("UTC") if end else pd.Timestamp.now(tz="UTC")
    t0 = pd.Timestamp(start, tz="UTC")
    ev = get("https://gamma-api.polymarket.com/events", slug=POLYMARKET_EVENT)[0]
    yes = [json.loads(m["clobTokenIds"])[0] for m in ev["markets"] if "D Senate, D House" in m["question"]][0]
    pts, a = {}, t0
    while a < end:                                   # Polymarket: at most 14 days per hourly request
        b = min(a + pd.Timedelta(days=14), end)
        for x in get("https://clob.polymarket.com/prices-history", market=yes, startTs=int(a.timestamp()),
                     endTs=int(b.timestamp()), fidelity=60)["history"]:
            if a.timestamp() <= x["t"] <= b.timestamp(): pts[x["t"]] = x["p"]
        a = b; time.sleep(0.3)
    poly = pd.Series(pts).sort_index(); poly.index = pd.to_datetime(poly.index, unit="s", utc=True).tz_convert(ET)
    k, a = {}, t0
    while a < end:                                   # Kalshi: hourly candles, 31 days per request
        b = min(a + pd.Timedelta(days=31), end)
        for x in get(f"{KAPI}/series/{KALSHI_SERIES}/markets/{KALSHI_EVENT}-DD/candlesticks", start_ts=int(a.timestamp()),
                     end_ts=int(b.timestamp()), period_interval=60)["candlesticks"]:
            bid, ask = x["yes_bid"].get("close_dollars"), x["yes_ask"].get("close_dollars")
            if bid and ask: k[x["end_period_ts"]] = (float(bid) + float(ask)) / 2
        a = b; time.sleep(0.3)
    kal = pd.Series(k).sort_index(); kal.index = pd.to_datetime(kal.index, unit="s", utc=True).tz_convert(ET)
    days = pd.bdate_range(t0.date(), end.tz_convert(ET).date())
    snap = lambda s: pd.Series({d.date(): s[:pd.Timestamp(d.date()).tz_localize(ET) + pd.Timedelta(hours=16)].iloc[-1]
                                for d in days if len(s[:pd.Timestamp(d.date()).tz_localize(ET) + pd.Timedelta(hours=16)])})
    out = pd.DataFrame({"poly_DD_4pm": snap(poly), "kalshi_DD_4pm": snap(kal)})
    out.index.name = "date"
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "hourly":
        h = hourly_4pm(); h.to_csv(DATA / "pm_4pm.csv"); print(h.tail().round(3), h.shape); sys.exit()
    p, pm = polymarket()
    k, km = kalshi()
    df = p.join(k, how="outer").sort_index()
    df.index.name = "date"
    df.to_csv(DATA / "pm_daily.csv")
    json.dump(pm + km, open(DATA / "pm_meta.json", "w"), indent=1)
    print(df.tail(8).round(3).to_string())
    print(df[["poly_DD", "kalshi_DD"]].describe().round(3))
