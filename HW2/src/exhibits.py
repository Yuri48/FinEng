"""Tables and figures for the report. Everything is written to outputs/."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from . import config as C
from .market import CHANGE_COLS, CHANGE_LABELS

TYPE_ORDER = ["statement", "minutes", "speech"]
SUBTYPE_LABEL = {"statement": "Statement", "minutes": "Minutes", "speech": "Speech",
                 "testimony": "Testimony", "press_conference": "Press conference"}
COLORS = {"statement": "#1f4e79", "minutes": "#c55a11", "speech": "#548235"}


def table1(corpus: pd.DataFrame) -> pd.DataFrame:
    t = corpus.pivot_table(index=["doc_type", "subtype"], columns="chair", values="doc_id", aggfunc="count", fill_value=0)
    t = t.reindex(columns=["Powell", "Warsh"])
    t["Total"] = t.sum(axis=1)
    words = corpus.groupby(["doc_type", "subtype"])["n_words"].median().rename("Median words")
    t = t.join(words)
    t = t.reset_index()
    t["doc_type"] = pd.Categorical(t["doc_type"], TYPE_ORDER)
    t = t.sort_values(["doc_type", "subtype"])
    t["Document"] = t["subtype"].map(SUBTYPE_LABEL)
    t = t[["Document", "Powell", "Warsh", "Total", "Median words"]]
    tot = pd.DataFrame([dict(Document="All documents", Powell=t.Powell.sum(), Warsh=t.Warsh.sum(), Total=t.Total.sum(),
                             **{"Median words": corpus.n_words.median()})])
    t = pd.concat([t, tot], ignore_index=True)
    t["Median words"] = t["Median words"].round(0).astype(int)
    t.to_csv(C.OUTPUTS / "table1_documents.csv", index=False)
    return t


def figure1(scores: pd.DataFrame, path=None) -> str:
    path = path or C.OUTPUTS / "figure1_tone.png"
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    warsh = pd.Timestamp(C.WARSH_START)
    for ax, s in zip(axes, C.TONE_SCORES):
        for dt in TYPE_ORDER:
            sub = scores[scores.doc_type == dt].sort_values("date")
            ax.plot(sub.date, sub[s], marker="o", ms=3, lw=0.8, alpha=0.45, color=COLORS[dt])
            roll = sub.set_index("date")[s].rolling("180D", min_periods=2).mean()
            ax.plot(roll.index, roll.values, lw=2.2, color=COLORS[dt], label=f"{dt} (6-month mean)")
        ax.axvline(warsh, color="black", ls="--", lw=1.2)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_ylabel(C.TONE_LABELS[s])
        ax.grid(alpha=0.25)
    axes[0].annotate("Warsh sworn in\n22 May 2026", xy=(warsh, axes[0].get_ylim()[1]), xytext=(-8, -4),
                     textcoords="offset points", ha="right", va="top", fontsize=9)
    axes[0].legend(loc="lower left", fontsize=8, ncol=3)
    axes[0].set_title("Figure 1. Hawkish (+) / dovish (−) tone of Fed communication by document type, Feb 2018 – Sep 2026")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def table2(df: pd.DataFrame) -> pd.DataFrame:
    w = df[df.chair == "Warsh"].sort_values("date").copy()
    out = pd.DataFrame({
        "Release": w.date.dt.strftime("%Y-%m-%d"),
        "Document": w.subtype.map(SUBTYPE_LABEL),
        "Title": w.title.str.slice(0, 48),
        "Word list": w.lex_net.round(2),
        "FinBERT sent.": w.fb_sent.round(3),
        "FinBERT sim.": (w.fb_sim * 100).round(2),
        "ΔDXY (%)": w.dDXY.round(2),
        "Δ10s2s (bp)": w.d10s2s.round(0),
        "Δ1y (bp)": w.dDGS1.round(0),
        "ΔG−V (%)": w.dGmV.round(2),
        "Δ3m bill (bp)": w.dDGS3MO.round(0),
    })
    out.to_csv(C.OUTPUTS / "table2_warsh_releases.csv", index=False)
    return out


def summary_by_chair(scores: pd.DataFrame) -> pd.DataFrame:
    g = scores.groupby(["doc_type", "chair"])[C.TONE_SCORES].agg(["mean", "std", "count"])
    g.to_csv(C.OUTPUTS / "tone_by_chair.csv")
    return g
