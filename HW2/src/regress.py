"""Market-reaction regressions.

For each indicator y and each tone score s:
    dy_i = a + b * z(s_i) + c * dDGS3MO_i + [document-type dummies] + e_i
with HC1 robust standard errors. z() standardises the tone score over the estimation
sample so b is the effect of a one-standard-deviation more hawkish document. The
3-month bill change absorbs the rate decision itself, so b is credited only with what
the words add.

Two versions of the regressor: the tone *level* and the tone *change* since the
previous document of the same type (the "surprise" proxy used by Doh, Song and Yang).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config as C
from .market import CHANGE_COLS


def add_tone_changes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    for s in C.TONE_SCORES:
        df[f"{s}_chg"] = df.groupby("subtype")[s].diff()
    return df


def _fit(y: pd.Series, X: pd.DataFrame):
    m = sm.OLS(y, sm.add_constant(X), missing="drop").fit(cov_type="HC1")
    return m


def run_table(df: pd.DataFrame, regressor_suffix: str = "", type_dummies: bool = True) -> pd.DataFrame:
    """One row per (indicator, tone score)."""
    rows = []
    for y in CHANGE_COLS:
        for s in C.TONE_SCORES:
            x = f"{s}{regressor_suffix}"
            sub = df[[y, x, "dDGS3MO", "doc_type"]].dropna()
            if len(sub) < 10:
                continue
            X = pd.DataFrame({"tone": (sub[x] - sub[x].mean()) / sub[x].std(ddof=0), "dDGS3MO": sub["dDGS3MO"]}, index=sub.index)
            if type_dummies and sub["doc_type"].nunique() > 1:
                X = pd.concat([X, pd.get_dummies(sub["doc_type"], prefix="type", drop_first=True, dtype=float)], axis=1)
            m = _fit(sub[y], X)
            rows.append(dict(indicator=y, tone=s,
                             b_tone=m.params["tone"], se_tone=m.bse["tone"], t_tone=m.tvalues["tone"], p_tone=m.pvalues["tone"],
                             b_bill=m.params["dDGS3MO"], t_bill=m.tvalues["dDGS3MO"],
                             n=int(m.nobs), r2=m.rsquared))
    return pd.DataFrame(rows)


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def format_table(tab: pd.DataFrame, labels: dict) -> pd.DataFrame:
    """Wide layout: rows = tone score, columns = indicator, cells = 'b (t)'."""
    out = {}
    for y in CHANGE_COLS:
        col = {}
        for _, r in tab[tab.indicator == y].iterrows():
            col[C.TONE_LABELS[r.tone]] = f"{r.b_tone:+.3f}{stars(r.p_tone)} ({r.t_tone:.2f})"
        out[labels[y]] = col
    return pd.DataFrame(out)
