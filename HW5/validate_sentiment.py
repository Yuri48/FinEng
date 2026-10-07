"""Validation of the sentiment measures against 200 hand-labelled documents (stratified: 50 discovery headlines,
25 issue headlines, 40 Bluesky, 30 Mastodon, 30 Lemmy, 25 Reddit), labelled by an LLM annotator (Claude) reading each
text: tone in {-1, 0, +1}; partisan direction in {-1, 0, +1} (+1 = favours Democrats or is critical of Trump/GOP).

Reported: Spearman correlation of each continuous score with the label; 3-class accuracy and macro-F1 for tone
(RoBERTa argmax; VADER with the standard +/-0.05 cut-offs); for partisan direction, sign accuracy on the documents the
annotator labelled non-zero, for the ABSA-based pro_dem score and for a naive alternative that flips document tone
by which party is mentioned. -> output/table_validation.csv
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from config import DATA, OUT
import sentiment as S


def macro_f1(y, p):
    f = []
    for c in (-1, 0, 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


if __name__ == "__main__":
    v = pd.read_csv(DATA / "validation_sample.csv")
    L = json.load(open(DATA / "validation_labels.json"))
    v["y_tone"] = [L["labels"][str(i)]["tone"] for i in v.index]
    v["y_part"] = [L["labels"][str(i)]["partisan"] for i in v.index]
    tone, pneg, ppos = S.roberta(v.text.tolist())
    v["tone_rob"] = tone; v["tone_vader"] = S.vader(v.text.tolist())
    pneu = 1 - pneg - ppos
    v["cls_rob"] = np.select([ppos >= np.maximum(pneg, pneu), pneg >= np.maximum(ppos, pneu)], [1, -1], 0)
    v["cls_vader"] = np.select([v.tone_vader >= 0.05, v.tone_vader <= -0.05], [1, -1], 0)
    v = S.partisan(v)
    gop, dem = v.gop_aspect.notna(), v.dem_aspect.notna()
    v["naive_pro_dem"] = np.where(gop & ~dem, -v.tone_rob, np.where(dem & ~gop, v.tone_rob, np.nan))
    rows = []
    for name, col, cls in (("RoBERTa tone", "tone_rob", "cls_rob"), ("VADER tone", "tone_vader", "cls_vader")):
        rows.append({"measure": name, "target": "tone", "n": len(v), "spearman": stats.spearmanr(v[col], v.y_tone)[0],
                     "accuracy_3class": (v[cls] == v.y_tone).mean(), "macro_f1": macro_f1(v.y_tone.values, v[cls].values),
                     "majority_baseline": v.y_tone.value_counts(normalize=True).max()})
    for name, col in (("ABSA pro-Democratic", "pro_dem"), ("Naive tone x party", "naive_pro_dem"), ("RoBERTa tone (as proxy)", "tone_rob")):
        d = v[v[col].notna()]
        nz = d[d.y_part != 0]
        sgn = -np.sign(nz[col]) if col == "tone_rob" else np.sign(nz[col])
        rows.append({"measure": name, "target": "partisan", "n": len(d), "spearman": stats.spearmanr(d[col], d.y_part)[0],
                     "sign_accuracy_nonzero": (sgn == nz.y_part).mean(), "n_nonzero": len(nz),
                     "coverage": len(d) / len(v)})
    t = pd.DataFrame(rows).set_index(["target", "measure"])
    t.round(3).to_csv(OUT / "table_validation.csv")
    v.drop(columns=["text"]).to_csv(OUT / "validation_scored.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string())
    print("off-topic share (exam 'midterms'):", len(L["off_topic"]) / len(v))
    print(pd.crosstab(v.y_tone, v.cls_rob, rownames=["label"], colnames=["RoBERTa"]))
