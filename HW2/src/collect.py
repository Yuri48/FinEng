"""Collect FOMC statements, minutes and the Chair's speeches, testimony and
press-conference transcripts from federalreserve.gov.

Every document is cached under data/raw/ so re-runs are free. The parsed corpus
(one row per document, with release date/time and clean text) goes to
data/interim/corpus.csv. Nothing under data/ is committed.
"""
from __future__ import annotations

import io
import json
import re
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from . import config as C

SESSION = requests.Session()
SESSION.headers["User-Agent"] = C.USER_AGENT


def _get(url: str, cache_name: str, binary: bool = False, pause: float = 0.4):
    """Fetch a URL with an on-disk cache. Returns text (or bytes) or None on 404."""
    path = C.RAW / cache_name
    if path.exists():
        return path.read_bytes() if binary else path.read_text(encoding="utf-8")
    for attempt in range(4):
        r = SESSION.get(url, timeout=60)
        if r.status_code == 404:
            return None
        if r.ok:
            break
        time.sleep(2 * (attempt + 1))
    r.raise_for_status()
    time.sleep(pause)
    r.encoding = "utf-8"          # federalreserve.gov serves UTF-8; requests would otherwise guess ISO-8859-1
    if binary:
        path.write_bytes(r.content)
        return r.content
    path.write_text(r.text, encoding="utf-8")
    return r.text


# --------------------------------------------------------------------------- HTML text

def _article_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    art = soup.find(id="article") or soup.find("div", class_="col-md-8") or soup.body
    for tag in art.find_all(["script", "style", "nav", "table", "sup"]):
        tag.decompose()
    # Drop footnotes, "Last update" and the "For media inquiries" block.
    paras = []
    for p in art.find_all(["p", "li", "h4", "h5"]):
        t = " ".join(p.get_text(" ", strip=True).split())
        if not t or len(t) < 3:
            continue
        low = t.lower()
        if low in ("share",) or low.startswith(("for media inquiries", "for release at", "for immediate release", "last update", "return to text", "implementation note issued", "1. ", "footnote")):
            continue
        paras.append(t)
    text = "\n".join(paras)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


# --------------------------------------------------------------------------- statements & minutes

def _calendar_pages() -> list[str]:
    pages = [_get(f"{C.FED}/monetarypolicy/fomccalendars.htm", "fomccalendars.htm")]
    for y in range(2018, 2021):
        pages.append(_get(f"{C.FED}/monetarypolicy/fomchistorical{y}.htm", f"fomchistorical{y}.htm"))
    return pages


def collect_statements() -> pd.DataFrame:
    rows = []
    seen = set()
    for page in _calendar_pages():
        for m in re.finditer(r"/newsevents/pressreleases/monetary(\d{8})a\.htm", page):
            d = m.group(1)
            if d in seen or d < "20180201":
                continue
            seen.add(d)
            url = f"{C.FED}/newsevents/pressreleases/monetary{d}a.htm"
            html = _get(url, f"statement_{d}.htm")
            if "issues FOMC statement" not in html:   # e.g. Statement on Longer-Run Goals, implementation notes
                continue
            text = _article_text(html)
            rows.append(dict(doc_type="statement", doc_id=f"statement_{d}",
                             date=pd.Timestamp(d), time=C.STATEMENT_TIME,
                             title="FOMC statement", url=url, text=text))
    df = pd.DataFrame(rows).sort_values("date")
    # The 15 March 2020 emergency statement went out on a Sunday at 5:00 PM.
    df.loc[df.doc_id == "statement_20200315", "time"] = "17:00"
    return df


def collect_minutes() -> pd.DataFrame:
    rows, seen = [], set()
    for page in _calendar_pages():
        # Each minutes link is followed on the page by "(Released Month DD, YYYY)".
        for m in re.finditer(r"fomcminutes(\d{8})\.htm(.{0,2500}?)Released\s+([A-Za-z]+ \d{1,2}, \d{4})", page, re.S):
            mtg, released = m.group(1), m.group(3)
            if mtg in seen or mtg < "20180101":
                continue
            seen.add(mtg)
            url = f"{C.FED}/monetarypolicy/fomcminutes{mtg}.htm"
            html = _get(url, f"minutes_{mtg}.htm")
            if html is None:
                continue
            rel = pd.Timestamp(datetime.strptime(released, "%B %d, %Y"))
            rows.append(dict(doc_type="minutes", doc_id=f"minutes_{mtg}",
                             date=rel, time=C.MINUTES_TIME, meeting=pd.Timestamp(mtg),
                             title=f"Minutes of the {pd.Timestamp(mtg):%B %Y} meeting", url=url,
                             text=_article_text(html)))
    return pd.DataFrame(rows).sort_values("date")


# --------------------------------------------------------------------------- Chair speeches, testimony

def _chair_at(date: pd.Timestamp) -> str:
    return "Warsh" if date >= pd.Timestamp(C.WARSH_START) else "Powell"


def _parse_dt(s: str) -> datetime:
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(s)


def _is_chair_item(item: dict) -> bool:
    """Keep only items delivered by the sitting Chair (not Governor Powell in 2012-17 or after May 2026)."""
    s = item.get("s", "")
    d = pd.Timestamp(_parse_dt(item["d"]))
    if d < pd.Timestamp(C.POWELL_START):
        return False
    if "Powell" in s and d < pd.Timestamp(C.WARSH_START) and "Governor" not in s:
        return True
    if "Warsh" in s and d >= pd.Timestamp(C.WARSH_START) and "Governor" not in s:
        return True
    return False


def _json_feed(name: str) -> list[dict]:
    raw = _get(f"{C.FED}/json/{name}.json", f"{name}.json")
    return [x for x in json.loads(raw.lstrip("﻿")) if "d" in x and "l" in x]


def collect_speeches_and_testimony() -> pd.DataFrame:
    rows = []
    for feed, kind in (("ne-speeches", "speech"), ("ne-testimony", "testimony")):
        for item in _json_feed(feed):
            if not _is_chair_item(item):
                continue
            dt = _parse_dt(item["d"])
            url = C.FED + item["l"]
            slug = item["l"].rsplit("/", 1)[-1].replace(".htm", "")
            html = _get(url, f"{kind}_{slug}.htm")
            if html is None:
                continue
            text = _article_text(html)
            if len(text.split()) < 150:      # short statements ("Acceptance remarks") carry no policy tone
                continue
            rows.append(dict(doc_type="speech", subtype=kind, doc_id=f"{kind}_{slug}",
                             date=pd.Timestamp(dt.date()), time=dt.strftime("%H:%M"),
                             title=item["t"], url=url, text=text))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- press conferences

_SPEAKER = re.compile(r"^(CHAIR(?:MAN)?\s+(?:POWELL|WARSH)\.|[A-Z][A-Z .'\-]{2,60}\.)\s", re.M)


def _chair_only(text: str) -> str:
    """Keep the Chair's own words from a press-conference transcript; drop reporters' questions."""
    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r"Page \d+ of \d+", " ", text)
    parts = _SPEAKER.split(text)
    if len(parts) < 3:
        return text
    keep, cur = [], None
    for chunk in parts:
        if _SPEAKER.match(chunk + " "):
            cur = chunk
            continue
        if cur and cur.startswith("CHAIR"):
            keep.append(chunk)
    out = " ".join(keep)
    return out if len(out.split()) > 500 else text


def collect_press_conferences(statement_dates) -> pd.DataFrame:
    from pypdf import PdfReader
    rows = []
    for d in statement_dates:
        ds = pd.Timestamp(d).strftime("%Y%m%d")
        url = f"{C.FED}/mediacenter/files/FOMCpresconf{ds}.pdf"
        pdf = _get(url, f"presconf_{ds}.pdf", binary=True)
        if pdf is None:
            continue
        reader = PdfReader(io.BytesIO(pdf))
        raw = "\n".join((pg.extract_text() or "") for pg in reader.pages)
        raw = re.sub(r"-\n", "", raw)                       # de-hyphenate line breaks
        raw = re.sub(r"(?<![.\n])\n(?![A-Z]{3})", " ", raw)  # join wrapped lines
        text = _chair_only(raw)
        rows.append(dict(doc_type="speech", subtype="press_conference", doc_id=f"presconf_{ds}",
                         date=pd.Timestamp(d), time=C.PRESSCONF_TIME,
                         title=f"Press conference, {pd.Timestamp(d):%d %B %Y}", url=url, text=text))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- corpus

def build_corpus(save: bool = True) -> pd.DataFrame:
    st = collect_statements()
    mi = collect_minutes()
    sp = collect_speeches_and_testimony()
    pc = collect_press_conferences(st.date)
    st["subtype"] = "statement"
    mi["subtype"] = "minutes"
    corpus = pd.concat([st, mi, sp, pc], ignore_index=True)
    corpus["date"] = pd.to_datetime(corpus["date"])
    corpus = corpus[(corpus.date >= C.POWELL_START) & (corpus.date <= C.SAMPLE_END)]
    corpus["chair"] = corpus.date.map(_chair_at)
    corpus["n_words"] = corpus.text.str.split().str.len()
    corpus = corpus.sort_values(["date", "doc_type"]).reset_index(drop=True)
    if save:
        corpus.to_csv(C.INTERIM / "corpus.csv", index=False)
    return corpus


def load_corpus() -> pd.DataFrame:
    df = pd.read_csv(C.INTERIM / "corpus.csv", parse_dates=["date", "meeting"])
    return df
