"""Score a recommender against held-out bids.

This module knows nothing about how recommendations are produced. It takes a
ranking function — supplier id in, ordered list of tender ids out — and scores
it, so every version of the system is judged by identical code.

Ground truth comes from the 70/30 split. A supplier's profile is built from the
tenders they bid on in the train half; the test half contains bids they placed
that the system was never shown. If a recommendation appears there, the system
predicted a real decision.

Run it directly to see the random baseline:

    python evaluate.py
"""

import random
import statistics

import pandas as pd

SAMPLE_DIR = "sample_data"

# A supplier needs enough held-out bids for precision@10 to mean anything.
# With 6 relevant items, one lucky hit moves the score by 17 points.
MIN_HELD_OUT = 10

# Cutoffs to report. Small k answers "is the top of the list any good",
# large k answers "does the right answer appear anywhere useful".
K_VALUES = (5, 10, 20, 50)


def load():
    """Return (participants, test tenders).

    participants: pn_lot, post_num, is_winner, fz  — one row per bid
    test:         the candidate pool, 10 columns of tender metadata
    """
    participants = pd.read_csv(f"{SAMPLE_DIR}/participants.csv", low_memory=False)
    test = pd.read_csv(f"{SAMPLE_DIR}/test.csv", low_memory=False)
    return participants, test


def build_ground_truth(participants, test):
    """Map each supplier to the set of test-split tenders they actually bid on.

    This is the entire definition of "correct" in this project. A bid is an
    action the supplier really took, on a tender the recommender was not shown
    when building their profile.

    Suppliers with too few held-out bids are dropped here rather than at
    reporting time, so they never enter any average.
    """
    test_lots = set(test["pn_lot"])
    in_test = participants[participants["pn_lot"].isin(test_lots)]

    truth = {}
    for supplier, rows in in_test.groupby("post_num"):
        lots = set(rows["pn_lot"])
        if len(lots) >= MIN_HELD_OUT:
            truth[supplier] = lots
    return truth


def precision_at_k(ranked, relevant, k):
    """Of the top k recommendations, what fraction were real bids.

    Answers: if a supplier looked at the first k suggestions, how much of what
    they saw was worth their time.
    """
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for lot in top if lot in relevant) / len(top)


def recall_at_k(ranked, relevant, k):
    """Of everything they actually bid on, what fraction surfaced in the top k.

    Answers the opposite question: how much did the system miss. A recommender
    can have perfect precision by returning one safe result and hiding
    everything else; recall is what catches that.
    """
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def evaluate(rank_fn, verbose=True):
    """Score a ranking function across all eligible suppliers.

    rank_fn(supplier_id, candidates) must return tender ids ordered best-first.
    `candidates` is the full test frame; the ranker decides what to do with it.

    Returns {"precision@k": mean, "recall@k": mean, ...} and prints a table
    with the spread, because a mean over 30 suppliers hides more than it shows.
    """
    participants, test = load()
    truth = build_ground_truth(participants, test)

    if not truth:
        raise SystemExit(
            "No supplier has enough held-out bids. The test split is missing "
            "the tenders these suppliers actually bid on — re-run make_sample.py."
        )

    # Per-supplier scores, kept individually so spread can be reported.
    scores = {f"precision@{k}": [] for k in K_VALUES}
    scores.update({f"recall@{k}": [] for k in K_VALUES})

    for supplier, relevant in truth.items():
        ranked = list(rank_fn(supplier, test))
        for k in K_VALUES:
            scores[f"precision@{k}"].append(precision_at_k(ranked, relevant, k))
            scores[f"recall@{k}"].append(recall_at_k(ranked, relevant, k))

    if verbose:
        held = [len(v) for v in truth.values()]
        print(f"suppliers scored : {len(truth)}")
        print(f"held-out bids    : {min(held)}–{max(held)} "
              f"(median {int(statistics.median(held))})")
        print(f"candidate pool   : {len(test):,} tenders\n")
        print(f"{'metric':<14}{'mean':>8}{'median':>9}{'min':>8}{'max':>8}")
        print("-" * 47)
        for name, values in scores.items():
            print(f"{name:<14}{statistics.mean(values):>8.1%}"
                  f"{statistics.median(values):>9.1%}"
                  f"{min(values):>8.1%}{max(values):>8.1%}")

    return {name: statistics.mean(values) for name, values in scores.items()}


def random_ranker(seed=42):
    """A ranker that ignores the supplier entirely — the floor.

    Any real system must beat this. If it doesn't, the signal it claims to use
    isn't there.
    """
    rng = random.Random(seed)

    def rank(supplier, candidates):
        lots = candidates["pn_lot"].tolist()
        rng.shuffle(lots)
        return lots

    return rank


if __name__ == "__main__":
    print("Baseline: random ranking\n")
    evaluate(random_ranker())
