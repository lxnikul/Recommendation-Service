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

# Candidate orderings to average over. Standard error falls as 1/sqrt(n), so
# 30 is already tight enough to compare configurations and costs a few seconds.
DEFAULT_REPEATS = 30


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


def evaluate(rank_fn, repeats=DEFAULT_REPEATS, verbose=True, seed=0):
    """Score a ranking function across all eligible suppliers.

    rank_fn(supplier_id, candidates) must return tender ids ordered best-first.
    `candidates` is the full test frame; the ranker decides what to do with it.

    Structured scores take few distinct values, so hundreds of candidates can
    tie at the top and the ten that get returned depend on how pandas' unstable
    sort happened to partition the rows. That moves precision@10 by up to nine
    points between runs of identical code.

    So each repeat shuffles the candidate order and the results are averaged,
    which removes the dependence on any one arbitrary ordering. The reported
    +/- is the standard error of that averaging — it says how reproducible the
    number is, NOT how well the system is characterised. Spread across
    suppliers is far larger and is what the min/max columns show.
    """
    participants, test = load()
    truth = build_ground_truth(participants, test)

    if not truth:
        raise SystemExit(
            "No supplier has enough held-out bids. The test split is missing "
            "the tenders these suppliers actually bid on — re-run make_sample.py."
        )

    metrics = ([f"precision@{k}" for k in K_VALUES]
               + [f"recall@{k}" for k in K_VALUES])

    # per_supplier[metric][supplier] -> one score per repeat
    per_supplier = {m: {s: [] for s in truth} for m in metrics}
    # per_run[metric] -> the across-supplier mean for each repeat
    per_run = {m: [] for m in metrics}

    for r in range(repeats):
        candidates = test if repeats == 1 else test.sample(
            frac=1.0, random_state=seed + r)

        run = {m: [] for m in metrics}
        for supplier, relevant in truth.items():
            ranked = list(rank_fn(supplier, candidates))
            for k in K_VALUES:
                p = precision_at_k(ranked, relevant, k)
                rc = recall_at_k(ranked, relevant, k)
                per_supplier[f"precision@{k}"][supplier].append(p)
                per_supplier[f"recall@{k}"][supplier].append(rc)
                run[f"precision@{k}"].append(p)
                run[f"recall@{k}"].append(rc)

        for m in metrics:
            per_run[m].append(statistics.mean(run[m]))

    # Each supplier's score is averaged over repeats first, so the spread shown
    # across suppliers is real variation between them rather than tie noise.
    supplier_means = {m: [statistics.mean(v) for v in per_supplier[m].values()]
                      for m in metrics}

    if verbose:
        held = [len(v) for v in truth.values()]
        print(f"suppliers scored : {len(truth)}")
        print(f"held-out bids    : {min(held)}–{max(held)} "
              f"(median {int(statistics.median(held))})")
        print(f"candidate pool   : {len(test):,} tenders")
        print(f"repeats          : {repeats}\n")
        print(f"{'metric':<14}{'mean':>8}{'±se':>7}{'median':>9}"
              f"{'min':>8}{'max':>8}")
        print("-" * 54)
        for m in metrics:
            values = supplier_means[m]
            se = (statistics.stdev(per_run[m]) / repeats ** 0.5
                  if repeats > 1 else 0.0)
            print(f"{m:<14}{statistics.mean(values):>8.1%}{se:>7.1%}"
                  f"{statistics.median(values):>9.1%}"
                  f"{min(values):>8.1%}{max(values):>8.1%}")

    return {m: statistics.mean(supplier_means[m]) for m in metrics}


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
    import argparse

    parser = argparse.ArgumentParser(description="Score the random baseline.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help="candidate orderings to average over")
    args = parser.parse_args()

    print("Baseline: random ranking\n")
    evaluate(random_ranker(), repeats=args.repeats)
