"""Step 2: score every document with the word list and with FinBERT; save data/interim/scores.csv."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src import config as C
from src.collect import load_corpus
from src.lexicon import score_corpus
from src.finbert import FinBERT

if __name__ == "__main__":
    corpus = load_corpus()
    lex = score_corpus(corpus)
    t0 = time.time()
    fb = FinBERT().score_corpus(corpus)
    print(f"FinBERT done in {time.time() - t0:.0f}s")
    scores = pd.concat([corpus.drop(columns=["text"]), lex.drop(columns=["n_words"]), fb], axis=1)
    scores.to_csv(C.INTERIM / "scores.csv", index=False)
    print(scores.groupby(["chair", "doc_type"])[["lex_net", "fb_sent", "fb_sim"]].mean().round(3))
