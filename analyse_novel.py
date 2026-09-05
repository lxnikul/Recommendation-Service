"""Measure ranking quality once obvious repeats are removed from the pool.

64% of correct answers sit at 0.90+ similarity to something the supplier has
already bid on, and 35% above 0.98. Public procurement repeats constantly, so
finding a returning tender is the most valuable thing this system does — but it
means the headline 74.8% is measured on a pool where a third of the wins are
the same lot coming round again.

The obvious follow-up is "how does it do on the rest", and the obvious way to
ask is wrong: restrict the answer key to novel bids and the familiar ones still
occupy the top slots, so the number craters for a reason unrelated to
performance — the system is being punished for correct answers that were
excluded from scoring.

So this strips repeats from the CANDIDATE POOL instead. A candidate is dropped
if its nearest match in the supplier's history exceeds the threshold, whether
or not it is a correct answer. That uses only history text and candidate text,
never labels, so nothing leaks. What remains is a pool of novel candidates
containing novel correct answers, and ordinary precision@10 applies — directly
comparable to the full-pool figure.

It is also a real deployment question. A supplier already tracking their annual
re-tenders wants to know what the system finds beyond them.

    python analyse_novel.py
    python analyse_novel.py --threshold 0.95 --semantic 0.3
"""

import argparse
import os

import numpy as np
import pandas as pd

import embed
import evaluate
import recommend

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")

THRESHOLD = 0.98
SEMANTIC_WEIGHT = 0.3
REPEATS = 10


def history_lots(participants, train, supplier):
    lots = set(participants.loc[participants["post_num"] == supplier, "pn_lot"])
    return train[train["pn_lot"].isin(lots)]["pn_lot"].tolist()


def novelty(candidates, history, cache):
    """Each candidate's similarity to the nearest tender in this history.

    Returned as a Series indexed by tender id, so callers join by id. A
    candidate with no vector gets 0 — unknown text is treated as novel rather
    than silently dropped.
    """
    hist_vecs, _ = cache.rows(history)
    cand_vecs, kept = cache.rows(candidates["pn_lot"].tolist())
    if len(hist_vecs) == 0 or len(cand_vecs) == 0:
        return pd.Series(0.0, index=candidates["pn_lot"])
    return pd.Series((cand_vecs @ hist_vecs.T).max(axis=1), index=kept)


def run(participants, train, test, cache, truth, threshold, w_semantic):
    """Precision@10 and recall@50 on the full pool and on the stripped pool."""
    rank = recommend.filters_ranker(participants, train, cache, w_semantic)

    full_p, full_r, cut_p, cut_r = [], [], [], []
    pool_sizes, key_sizes, dropped_correct = [], [], []

    for supplier, relevant in truth.items():
        history = history_lots(participants, train, supplier)
        sims = novelty(test, history, cache)

        # Label-free filter: drop candidates too close to anything in history.
        keep = test["pn_lot"].map(sims).fillna(0.0) < threshold
        pool = test[keep]
        novel_key = relevant & set(pool["pn_lot"])

        pool_sizes.append(len(pool))
        key_sizes.append(len(novel_key))
        dropped_correct.append(len(relevant) - len(novel_key))

        if not novel_key:
            continue

        fp = fr = cp = cr = 0.0
        for r in range(REPEATS):
            shuffled_full = test.sample(frac=1.0, random_state=r)
            shuffled_cut = pool.sample(frac=1.0, random_state=r)

            ranked_full = rank(supplier, shuffled_full)
            ranked_cut = rank(supplier, shuffled_cut)

            fp += evaluate.precision_at_k(ranked_full, relevant, 10)
            fr += evaluate.recall_at_k(ranked_full, relevant, 50)
            cp += evaluate.precision_at_k(ranked_cut, novel_key, 10)
            cr += evaluate.recall_at_k(ranked_cut, novel_key, 50)

        full_p.append(fp / REPEATS)
        full_r.append(fr / REPEATS)
        cut_p.append(cp / REPEATS)
        cut_r.append(cr / REPEATS)

    return {
        "n_suppliers": len(full_p),
        "pool": np.mean(pool_sizes),
        "key": np.mean(key_sizes),
        "dropped": np.mean(dropped_correct),
        "full_p": np.mean(full_p), "full_r": np.mean(full_r),
        "cut_p": np.mean(cut_p), "cut_r": np.mean(cut_r),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="similarity above which a candidate counts as a repeat")
    parser.add_argument("--semantic", type=float, default=SEMANTIC_WEIGHT)
    args = parser.parse_args()

    participants = pd.read_csv(f"{SAMPLE_DIR}/participants.csv", low_memory=False)
    train = pd.read_csv(f"{SAMPLE_DIR}/train.csv", low_memory=False)
    test = pd.read_csv(f"{SAMPLE_DIR}/test.csv", low_memory=False)
    cache = embed.load_tenders()
    truth = evaluate.build_ground_truth(participants, test)

    res = run(participants, train, test, cache, truth,
              args.threshold, args.semantic)

    print(f"semantic weight {args.semantic}, repeat threshold {args.threshold}")
    print(f"{res['n_suppliers']} suppliers had novel bids left to find\n")
    print(f"{'':<22}{'candidates':>12}{'answer key':>12}"
          f"{'prec@10':>10}{'recall@50':>11}")
    print("-" * 67)
    print(f"{'full pool':<22}{len(test):>12,}"
          f"{sum(len(v) for v in truth.values()) / len(truth):>12.0f}"
          f"{res['full_p']:>10.1%}{res['full_r']:>11.1%}")
    print(f"{'repeats stripped':<22}{res['pool']:>12,.0f}{res['key']:>12.0f}"
          f"{res['cut_p']:>10.1%}{res['cut_r']:>11.1%}")
    print()
    print(f"per supplier, stripping removed {res['dropped']:.0f} correct answers "
          f"and left {res['key']:.0f}")
