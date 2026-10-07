"""Step 1: what is the midterm conversation about? Topic discovery with BERTopic on every document that mentions the
midterms (news headlines + Bluesky + Mastodon + Reddit + Lemmy), January to October 2026.

    embeddings  sentence-transformers/all-mpnet-base-v2 (cached by doc_id in data/emb_*.npy)
    clustering  UMAP (5 dims, cosine) -> HDBSCAN; outliers reassigned to the nearest topic centroid if cos >= 0.40
    keywords    class-based TF-IDF on unigrams+bigrams, with MMR for diversity

The fine-grained clusters are then labelled and grouped into voter issues in data/topic_labels.json, written by an LLM
(Claude) that reads, for every cluster, the c-TF-IDF keywords and eight documents (output/topic_cards.md) - the
'LLM as topic labeller' step BERTopic supports. issues.py applies that mapping.
-> data/doc_topics.csv, output/topic_info.csv, output/topic_cards.md
"""
import json, sys
import numpy as np
import pandas as pd
from config import DATA, OUT
from corpus import load_corpus

EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"
SEED = 42
EXTRA_STOP = """midterm midterms election elections 2026 just like don people think going know really ve ll re amp new
said says say year years time make way want need right good did does got get gets lot thing things sure yes yeah
actually probably maybe way let isn doesn didn won wouldn shouldn ll ve im it's that's they're we're you're
""".split()


def embed(df):
    """Embeddings for every doc, cached by doc_id so reruns after more data arrives only encode the new documents."""
    f_ids, f_emb = DATA / "emb_ids.json", DATA / "emb_mpnet.npy"
    have_ids, have = ([], np.zeros((0, 768), dtype=np.float32))
    if f_ids.exists():
        have_ids, have = json.load(open(f_ids)), np.load(f_emb)
    pos = {d: i for i, d in enumerate(have_ids)}
    todo = df[~df.doc_id.isin(pos)]
    if len(todo):
        from sentence_transformers import SentenceTransformer
        import torch
        m = SentenceTransformer(EMB_MODEL, device="mps" if torch.backends.mps.is_available() else "cpu")
        new = m.encode(todo.text.tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True).astype(np.float32)
        have_ids, have = have_ids + todo.doc_id.tolist(), np.vstack([have, new])
        json.dump(have_ids, open(f_ids, "w")); np.save(f_emb, have)
        pos = {d: i for i, d in enumerate(have_ids)}
    return have[[pos[d] for d in df.doc_id]]


def fit(df, emb, min_cluster_size):
    from bertopic import BERTopic
    from bertopic.representation import MaximalMarginalRelevance
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    from umap import UMAP

    umap = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED)
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10, metric="euclidean",
                  cluster_selection_method="eom", prediction_data=True)
    vec = CountVectorizer(stop_words=list(ENGLISH_STOP_WORDS.union(EXTRA_STOP)), ngram_range=(1, 2), min_df=5,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'\-]+\b")
    tm = BERTopic(umap_model=umap, hdbscan_model=hdb, vectorizer_model=vec,
                  ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
                  representation_model={"MMR": MaximalMarginalRelevance(diversity=0.3, top_n_words=12)},
                  top_n_words=15, calculate_probabilities=False, verbose=True)
    docs = df.text.tolist()
    topics, _ = tm.fit_transform(docs, emb)
    n_out = int(np.sum(np.array(topics) == -1))
    # Reassign HDBSCAN noise to the closest topic when the match is clear; the rest stays unassigned (-1).
    new = tm.reduce_outliers(docs, topics, strategy="embeddings", embeddings=emb, threshold=0.40)
    tm.update_topics(docs, topics=new, vectorizer_model=vec,
                     ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
                     representation_model={"MMR": MaximalMarginalRelevance(diversity=0.3, top_n_words=12)}, top_n_words=15)
    print(f"HDBSCAN outliers {n_out} ({n_out/len(docs):.1%}) -> after reassignment {int(np.sum(np.array(new) == -1))}")
    return tm, np.array(new)


def cards(df, tm, k_docs=6):
    """One card per topic for the labelling step: size, source mix, keywords, representative and random documents."""
    info = tm.get_topic_info()
    rng = np.random.default_rng(SEED)
    lines, rows = [], []
    for _, r in info.iterrows():
        t = r.Topic
        sub = df[df.topic == t]
        mix = sub.source.value_counts(normalize=True).round(2).to_dict()
        words = [w for w, _ in tm.get_topic(t)][:15]
        mmr = [w for w, _ in (tm.get_topic(t, full=True).get("MMR") or [])][:12] if t != -1 else []
        reps = (tm.get_representative_docs(t) or [])[:3] if t != -1 else []
        rnd = sub.text.iloc[rng.permutation(len(sub))[:k_docs - len(reps)]].tolist()
        rows.append({"topic": t, "n": len(sub), "share": len(sub) / len(df), "social_share_of_topic": (sub.kind == "social").mean(),
                     "words": ", ".join(words), "mmr": ", ".join(mmr), "mix": json.dumps(mix)})
        lines.append(f"### Topic {t}  (n={len(sub)}, {len(sub)/len(df):.2%}; mix {mix})\n**c-TF-IDF:** {', '.join(words)}  \n"
                     f"**MMR:** {', '.join(mmr)}\n" + "\n".join(f"- {d[:180]}" for d in reps + rnd) + "\n")
    pd.DataFrame(rows).to_csv(OUT / "topic_info.csv", index=False)
    open(OUT / "topic_cards.md", "w").write("\n".join(lines))


def assign_new(threshold=0.40):
    """Assign documents that were not in the fitted model to the fitted topics by nearest topic centroid in embedding
    space (cosine >= threshold, else -1): the rule the fit itself uses for HDBSCAN outliers. Keywords stay as fitted;
    topic sizes and source mixes are recomputed. Used once: a loader bug had left the reverse Bluesky worker's file
    (16 Jul - 5 Oct) out of the fit."""
    df = load_corpus()
    old = pd.read_parquet(DATA / "corpus.parquet")
    new = df[~df.doc_id.isin(old.doc_id)].copy()
    E_old, E_new = embed(old), embed(new)
    ids = sorted(t for t in old.topic.unique() if t != -1)
    C = np.vstack([E_old[(old.topic == t).values].mean(0) for t in ids])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    sim = E_new @ C.T
    best = sim.argmax(1)
    new["topic"] = np.where(sim.max(1) >= threshold, np.array(ids)[best], -1)
    print(f"assigned {len(new)} new documents; {np.mean(new.topic == -1):.1%} below the threshold")
    allc = pd.concat([old, new[old.columns]], ignore_index=True)
    allc.to_parquet(DATA / "corpus.parquet")
    allc[["doc_id", "source", "kind", "date", "week", "topic"]].to_csv(DATA / "doc_topics.csv", index=False)
    info = pd.read_csv(OUT / "topic_info.csv")
    g = allc.groupby("topic")
    info["n"] = info.topic.map(g.size()); info["share"] = info.n / len(allc)
    info["social_share_of_topic"] = info.topic.map(g.kind.apply(lambda k: (k == "social").mean()))
    info["mix"] = info.topic.map(g.source.apply(lambda x: json.dumps(x.value_counts(normalize=True).round(2).to_dict())))
    info.to_csv(OUT / "topic_info.csv", index=False)
    return new


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "assign":
    assign_new(); sys.exit()

if __name__ == "__main__":
    df = load_corpus()
    print(df.source.value_counts().to_dict(), len(df))
    emb = embed(df)
    mcs = int(sys.argv[1]) if len(sys.argv) > 1 else max(40, len(df) // 600)
    tm, topics = fit(df, emb, mcs)
    df["topic"] = topics
    df[["doc_id", "source", "kind", "date", "week", "topic"]].to_csv(DATA / "doc_topics.csv", index=False)
    df.to_parquet(DATA / "corpus.parquet")
    cards(df, tm)
    print(tm.get_topic_info().head(40)[["Topic", "Count", "Name"]].to_string())
