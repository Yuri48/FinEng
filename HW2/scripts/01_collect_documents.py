"""Step 1: download statements, minutes, speeches, testimony and press conferences (Feb 2018 - today)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.collect import build_corpus

if __name__ == "__main__":
    corpus = build_corpus()
    print(corpus.groupby(["chair", "doc_type", "subtype"]).size())
    print(corpus[["doc_type", "n_words"]].groupby("doc_type").describe())
