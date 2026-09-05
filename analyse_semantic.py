"""Work out where the semantic layer's gain actually comes from.

Adding medoid similarity moved precision@10 from 43% to 75%. A jump that size
looks suspicious. There's plenty of way it can lie under the hood. 

The train/test split is random over tenders, not chronological. Roughly a
thousand tenders in this dataset share a title with another tender — the same
procurement re-run. If a supplier bid on one copy in the train half and its
twin sits in the test half, the twin's vector is identical, similarity is 1.0,
and it ranks first. That is a correct answer, and repeat procurement is fairly a real
thing to predict, but it is duplicate matching rather than semantic
understanding, and a chronological split would realistically show less of it.

So this script asks (and answers) three questions:

  1. How many held-out bids have a title the supplier has literally seen before?
  2. For the rest, how close is the nearest thing in their history? A tight
     cluster near 1.0 means duplicates; a spread across 0.6-0.9 means the
     encoder is generalising.
  3. Which suppliers gained, and does that match the prediction that the
     semantic layer helps generalists and does nothing for specialists?

    python analyse_semantic.py
"""

import os

import numpy as np
import pandas as pd

import embed
import evaluate
import recommend
import supplier_profile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")

SEMANTIC_WEIGHT = 0.3
REPEATS = 10          # enough for per-supplier comparison; 30 is for headlines
DUPLICATE_THRESHOLD = 0.98


def load():
    participants = pd.read_csv(f"{SAMPLE_DIR}/participants.csv", low_memory=False)
    train = pd.read_csv(f"{SAMPLE_DIR}/train.csv", low_memory=False)
    test = pd.read_csv(f"{SAMPLE_DIR}/test.csv", low_memory=False)
    return participants, train, test, embed.load_tenders()


def duplicate_report(participants, train, test, truth):
    """Questions 1 and 2: how much of the answer key is already in history."""
    train_title = dict(zip(train["pn_lot"], train["purchase_name"].astype(str)))
    test_title = dict(zip(test["pn_lot"], test["purchase_name"].astype(str)))

    exact = total = 0
    for supplier, relevant in truth.items():
        lots = set(participants.loc[participants["post_num"] == supplier, "pn_lot"])
        seen = {train_title[l].strip() for l in lots if l in train_title}
        for bid in relevant:
            total += 1
            exact += test_title.get(bid, "").strip() in seen

    print("1. Held-out bids whose title already appears verbatim in history")
    print(f"   {exact:,} of {total:,}  ({exact / total:.1%})\n")
    return exact, total


def similarity_report(participants, train, cache, truth):
    """How close is each correct answer to the nearest thing in history."""
    scores = []
    for supplier, relevant in truth.items():
        lots = set(participants.loc[participants["post_num"] == supplier, "pn_lot"])
        history = train[train["pn_lot"].isin(lots)]["pn_lot"].tolist()

        hist_vecs, _ = cache.rows(history)
        bid_vecs, _ = cache.rows(list(relevant))
        if len(hist_vecs) == 0 or len(bid_vecs) == 0:
            continue
        scores.extend((bid_vecs @ hist_vecs.T).max(axis=1).tolist())

    s = np.array(scores)
    print("2. Similarity of each correct answer to its nearest history tender")
    print(f"   n = {len(s):,}")
    for label, lo, hi in [("near-identical (>0.98)", 0.98, 1.01),
                          ("very close   (0.90-0.98)", 0.90, 0.98),
                          ("close        (0.75-0.90)", 0.75, 0.90),
                          ("moderate     (0.50-0.75)", 0.50, 0.75),
                          ("distant      (<0.50)", -1.0, 0.50)]:
        n = int(((s >= lo) & (s < hi)).sum())
        bar = "#" * int(40 * n / len(s))
        print(f"   {label:<26}{n:>6}  {n / len(s):>6.1%}  {bar}")
    print(f"   median {np.median(s):.3f}\n")
    return s


def per_supplier(participants, train, test, cache, truth):
    """Question 3: who gained, and does it match the prediction."""
    plain = recommend.filters_ranker(participants, train)
    smart = recommend.filters_ranker(participants, train, cache, SEMANTIC_WEIGHT)

    rows = []
    for supplier, relevant in truth.items():
        prof = supplier_profile.build_profile(supplier, participants, train, cache)

        p_plain = p_smart = 0.0
        for r in range(REPEATS):
            shuffled = test.sample(frac=1.0, random_state=r)
            p_plain += evaluate.precision_at_k(plain(supplier, shuffled), relevant, 10)
            p_smart += evaluate.precision_at_k(smart(supplier, shuffled), relevant, 10)

        rows.append({
            "supplier": supplier,
            "cats": prof.n_categories,
            "medoids": len(prof),
            "history": prof.n_history,
            "plain": p_plain / REPEATS,
            "smart": p_smart / REPEATS,
        })

    df = pd.DataFrame(rows)
    df["delta"] = df["smart"] - df["plain"]
    df = df.sort_values("delta", ascending=False)

    print("3. Per-supplier precision@10, structured only vs + semantic")
    print(f"   {'supplier':>10}{'cats':>6}{'hist':>6}{'plain':>8}{'smart':>8}{'delta':>8}")
    print("   " + "-" * 46)
    for _, r in df.iterrows():
        print(f"   {int(r.supplier):>10}{int(r.cats):>6}{int(r.history):>6}"
              f"{r.plain:>8.0%}{r.smart:>8.0%}{r.delta:>+8.0%}")

    narrow = df[df["cats"] <= 3]
    broad = df[df["cats"] >= 6]
    print("   " + "-" * 46)
    print(f"   specialists (<=3 categories, n={len(narrow)}): "
          f"{narrow['plain'].mean():.0%} -> {narrow['smart'].mean():.0%} "
          f"({narrow['delta'].mean():+.0%})")
    print(f"   generalists (>=6 categories, n={len(broad)}): "
          f"{broad['plain'].mean():.0%} -> {broad['smart'].mean():.0%} "
          f"({broad['delta'].mean():+.0%})")
    return df


if __name__ == "__main__":
    participants, train, test, cache = load()
    truth = evaluate.build_ground_truth(participants, test)
    print(f"{len(truth)} suppliers, {sum(len(v) for v in truth.values()):,} "
          f"held-out bids, {len(test):,} candidates\n")

    duplicate_report(participants, train, test, truth)
    similarity_report(participants, train, cache, truth)
    per_supplier(participants, train, test, cache, truth)
