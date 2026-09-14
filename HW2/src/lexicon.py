"""A word list built for monetary-policy language.

Each entry is a regular expression over lower-cased text. Phrases are grouped by
theme (inflation, activity, labor, policy) so the report can show which theme
drives a document's score. A "not"/"no"/"never" within the three words before a
match flips its sign, so "inflation has not eased" counts as hawkish.

Score:  lex_net = (hawkish - dovish) / (hawkish + dovish)      in [-1, 1]
        lex_per_1k = (hawkish - dovish) per 1,000 words
"""
from __future__ import annotations

import re

import pandas as pd

HAWKISH = {
    "inflation": [
        r"higher inflation", r"rising inflation", r"inflation (?:has |have )?(?:risen|increased|picked up|moved up|accelerated|firmed)",
        r"inflation (?:remains?|is|stays?) (?:elevated|high|too high|persistent|sticky|above)",
        r"elevated inflation", r"persistent inflation", r"inflation pressures?", r"price pressures?",
        r"upside risks? to inflation", r"inflation (?:is )?(?:well )?above (?:the |our )?(?:2 percent |two percent )?(?:objective|goal|target)",
        r"inflation expectations (?:have |has )?(?:risen|moved up|increased|drifted (?:up|higher))", r"unanchor",
        r"overheat", r"too hot", r"inflation risks?", r"inflation (?:is |remains )?(?:a )?(?:problem|choice)",
        r"reinflat", r"second[- ]round effects", r"inflation surprised? to the upside",
    ],
    "activity": [
        r"(?:solid|strong|robust|brisk|vigorous) (?:pace|rate|growth|gains?|expansion)", r"strong (?:economic )?(?:activity|demand|spending)",
        r"economy (?:is |remains )?strong", r"growth (?:has |is )?(?:strong|robust|solid|picked up|accelerated)",
        r"expanding at a (?:solid|strong|robust) (?:pace|rate)", r"upside risks? to (?:growth|activity|the outlook)",
        r"above[- ]trend", r"resilient",
    ],
    "labor": [
        r"tight labor market", r"labor market (?:remains?|is|has been) (?:tight|strong|very strong|robust)",
        r"strong (?:job|payroll|employment) (?:gains|growth)", r"low unemployment", r"unemployment (?:rate )?(?:remains?|is|has been) low",
        r"wage (?:growth|pressures?|gains) (?:remains?|is|are|has been)? ?(?:strong|elevated|robust|high|firm)", r"labor shortages?",
    ],
    "policy": [
        r"rais(?:e|ed|ing) (?:the )?(?:target range|interest rates?|the federal funds rate|rates?|policy rate)", r"increas(?:e|ed|ing) (?:in )?(?:the )?target range",
        r"rate (?:hikes?|increases?)", r"hik(?:e|es|ed|ing)", r"tighten(?:ing|ed)?(?: of)? (?:monetary )?policy", r"policy tightening", r"tighter (?:monetary )?policy",
        r"restrictive", r"less accommodative", r"remov(?:e|al|ing) (?:of )?(?:policy )?accommodation", r"withdraw(?:al of|ing)? (?:policy )?accommodation",
        r"firming", r"normaliz(?:e|ation|ing)", r"balance sheet (?:runoff|reduction|normalization)", r"reduc(?:e|ing|tion of) (?:the size of )?(?:its |the |our )?(?:securities )?holdings",
        r"vigilant", r"further (?:rate )?increases?", r"additional (?:policy )?firming", r"higher for longer", r"price stability (?:is|remains) (?:our|the) (?:priority|objective|first)",
        r"deliver price stability", r"restore price stability",
    ],
}

DOVISH = {
    "inflation": [
        r"lower inflation", r"inflation (?:has |have )?(?:eased|declined|moderated|slowed|come down|fallen|receded|diminished|cooled|softened|subsided)",
        r"inflation (?:remains?|is|stays?) (?:low|subdued|muted|contained|below|soft)", r"subdued inflation", r"muted inflation", r"disinflation",
        r"downside risks? to inflation", r"inflation (?:is |remains )?(?:running )?below (?:the |our )?(?:2 percent |two percent )?(?:objective|goal|target)",
        r"inflation expectations (?:remain|are|have remained|appear) (?:well )?anchored", r"progress (?:on|toward) (?:lower )?inflation", r"inflation (?:has )?(?:made|shown) progress",
        r"inflation (?:is|was) (?:transitory|temporary)", r"inflation (?:is )?(?:moving|coming) (?:back )?(?:down|toward)",
    ],
    "activity": [
        r"(?:slow|weak|soft|sluggish|subdued|modest|tepid) (?:pace|rate|growth|gains?|expansion|activity|demand|spending)",
        r"growth (?:has |is )?(?:slowed|weakened|softened|moderated|cooled|stalled)", r"economy (?:has |is )?(?:slowed|weakened|slowing|weakening)",
        r"downside risks? to (?:growth|activity|the outlook|the economy)", r"recession", r"downturn", r"contract(?:ion|ed|ing)", r"deteriorat",
        r"uncertaint(?:y|ies) (?:has |have )?(?:increased|risen|remains? (?:elevated|high))", r"headwinds", r"below[- ]trend", r"weakness",
    ],
    "labor": [
        r"labor market (?:has |is |remains )?(?:cooled|cooling|softened|softening|weakened|weakening|loosened|eased|easing|slack)",
        r"unemployment (?:rate )?(?:has )?(?:risen|increased|moved up|edged up|ticked up|climbed|is rising)", r"rising unemployment", r"higher unemployment",
        r"job losses", r"layoffs", r"labor market slack", r"slack in the labor market", r"(?:weak|soft|slowing|slower) (?:job|payroll|employment) (?:gains|growth)",
        r"downside risks? to employment", r"maximum employment (?:is|remains) (?:our|the) (?:priority|objective|first)",
    ],
    "policy": [
        r"lower(?:ed|ing)? (?:the )?(?:target range|interest rates?|the federal funds rate|rates?|policy rate)", r"reduc(?:e|ed|ing|tion in|tion of) (?:the )?target range",
        r"rate (?:cuts?|reductions?)", r"cut(?:s|ting)? (?:interest )?rates?", r"eas(?:e|ed|ing) (?:of )?(?:monetary )?policy", r"policy easing", r"easier (?:monetary )?policy",
        r"accommodat(?:ive|ion)", r"stimulus", r"support(?:ive)? (?:of |for )?(?:the )?(?:economy|recovery|economic activity)", r"asset purchases?", r"purchas(?:e|es|ing) (?:of )?(?:treasury|agency|securities)",
        r"balance sheet (?:expansion|growth)", r"further (?:rate )?(?:cuts?|reductions?)", r"additional (?:policy )?(?:easing|accommodation)",
        r"patient", r"act as appropriate to sustain the expansion", r"insurance cut", r"quantitative easing",
    ],
}

_NEG = re.compile(r"\b(?:not|no|never|neither|nor)\b")


def _compile(groups):
    return {k: [re.compile(p) for p in v] for k, v in groups.items()}


_H = _compile(HAWKISH)
_D = _compile(DOVISH)


def _count(text: str, patterns) -> tuple[int, int]:
    """Return (matches, negated matches) over all patterns."""
    n = neg = 0
    for pat in patterns:
        for m in pat.finditer(text):
            n += 1
            pre = text[max(0, m.start() - 40):m.start()]
            if _NEG.search(" ".join(pre.split()[-3:])):
                neg += 1
    return n, neg


def score_text(text: str) -> dict:
    t = " ".join(str(text).lower().split())
    n_words = max(len(t.split()), 1)
    out = {"n_words": n_words}
    hawk = dove = 0
    for theme in HAWKISH:
        h, hn = _count(t, _H[theme])
        d, dn = _count(t, _D[theme])
        # a negated hawkish phrase counts as dovish and vice versa
        h_eff = (h - hn) + dn
        d_eff = (d - dn) + hn
        out[f"hawk_{theme}"] = h_eff
        out[f"dove_{theme}"] = d_eff
        hawk += h_eff
        dove += d_eff
    out["hawk"] = hawk
    out["dove"] = dove
    out["lex_net"] = (hawk - dove) / (hawk + dove) if (hawk + dove) else 0.0
    out["lex_per_1k"] = (hawk - dove) / n_words * 1000
    return out


def score_corpus(corpus: pd.DataFrame) -> pd.DataFrame:
    rows = [score_text(t) for t in corpus["text"]]
    return pd.DataFrame(rows, index=corpus.index)


def n_patterns() -> dict:
    return {"hawkish": sum(len(v) for v in HAWKISH.values()), "dovish": sum(len(v) for v in DOVISH.values())}
