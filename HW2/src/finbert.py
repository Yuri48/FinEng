"""FinBERT (ProsusAI/finbert) tone scores, two ways, from one forward pass per sentence.

fb_sent : mean over sentences of P(positive) - P(negative) from FinBERT's sentiment head.
          Positive = optimistic about activity / prices, which in Fed-speak leans hawkish,
          but this is a sentiment measure, not a policy-direction measure.
fb_sim  : mean over sentences of  cos(s, hawkish anchors) - cos(s, dovish anchors),
          using mean-pooled last-layer FinBERT embeddings. Anchors are short key
          sentences such as "Interest rates will rise". This measures policy direction.

Sentence scores are cached per document under data/interim/finbert/ so re-runs are free.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import config as C

MODEL = "ProsusAI/finbert"
CACHE = C.INTERIM / "finbert"
CACHE.mkdir(exist_ok=True)

HAWKISH_ANCHORS = [
    "Interest rates will rise.",
    "The Committee decided to raise the target range for the federal funds rate.",
    "Inflation remains elevated and further tightening of monetary policy will be appropriate.",
    "The labor market is tight and wage pressures are strong.",
    "Economic activity has been expanding at a strong pace.",
    "Upside risks to inflation have increased.",
    "The Committee will act to restore price stability.",
    "Monetary policy will need to be more restrictive.",
]
DOVISH_ANCHORS = [
    "Interest rates will fall.",
    "The Committee decided to lower the target range for the federal funds rate.",
    "Inflation has eased and additional policy accommodation will be appropriate.",
    "The labor market has softened and unemployment has risen.",
    "Economic activity has slowed.",
    "Downside risks to the outlook have increased.",
    "The Committee will act to support the economy.",
    "Monetary policy will need to be more accommodative.",
]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“(])")


def split_sentences(text: str, min_words: int = 4, max_words: int = 120) -> list[str]:
    text = re.sub(r"\s+", " ", str(text)).strip()
    out = []
    for s in _SENT_SPLIT.split(text):
        n = len(s.split())
        if min_words <= n <= max_words:
            out.append(s.strip())
    return out


class FinBERT:
    def __init__(self, device: str | None = None, batch_size: int = 32, max_length: int = 128):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL, output_hidden_states=True)
        self.model.to(self.device).eval()
        self.bs, self.max_length = batch_size, max_length
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        self.i_pos, self.i_neg = self.labels.index("positive"), self.labels.index("negative")
        self.hawk = self._embed(HAWKISH_ANCHORS)[1].mean(0)
        self.dove = self._embed(DOVISH_ANCHORS)[1].mean(0)
        self.hawk /= np.linalg.norm(self.hawk)
        self.dove /= np.linalg.norm(self.dove)

    @torch.no_grad()
    def _embed(self, sents: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return (probabilities [n,3], unit-norm mean-pooled embeddings [n,768])."""
        probs, embs = [], []
        for i in range(0, len(sents), self.bs):
            batch = sents[i:i + self.bs]
            enc = self.tok(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            out = self.model(**enc)
            p = torch.softmax(out.logits, -1).float().cpu().numpy()
            h = out.hidden_states[-1]
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * mask).sum(1) / mask.sum(1)
            e = torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy()
            probs.append(p)
            embs.append(e)
        return np.vstack(probs), np.vstack(embs)

    def score_document(self, doc_id: str, text: str) -> dict:
        cache = CACHE / f"{doc_id}.json"
        if cache.exists():
            return json.loads(cache.read_text())
        sents = split_sentences(text)
        if not sents:
            res = dict(n_sent=0, fb_sent=np.nan, fb_sim=np.nan, fb_pos=np.nan, fb_neg=np.nan)
        else:
            p, e = self._embed(sents)
            sim = e @ self.hawk - e @ self.dove
            res = dict(n_sent=len(sents),
                       fb_sent=float((p[:, self.i_pos] - p[:, self.i_neg]).mean()),
                       fb_pos=float(p[:, self.i_pos].mean()), fb_neg=float(p[:, self.i_neg].mean()),
                       fb_sim=float(sim.mean()),
                       fb_sim_top=float(np.sort(sim)[-max(1, len(sim) // 10):].mean() + np.sort(sim)[:max(1, len(sim) // 10)].mean()))
        cache.write_text(json.dumps(res))
        return res

    def score_corpus(self, corpus: pd.DataFrame, log_every: int = 25) -> pd.DataFrame:
        rows = []
        for k, (doc_id, text) in enumerate(zip(corpus["doc_id"], corpus["text"])):
            rows.append(self.score_document(doc_id, text))
            if log_every and (k + 1) % log_every == 0:
                print(f"  finbert {k + 1}/{len(corpus)}", flush=True)
        return pd.DataFrame(rows, index=corpus.index)
