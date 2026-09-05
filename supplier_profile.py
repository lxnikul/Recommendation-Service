"""Build a supplier profile: a few real past tenders that stand for their history.

A supplier who bids different categories cannot be represented by one
averaged vector — the average lands in a region of the space where no tender
exists, and matches everything equally. So the profile is a set of
medoids, for each thing the supplier does, the one real past tender most
typical of it.

Grouping is by ОКПД2 code rather than by clustering the vectors. The codes are
already there, the groups come out named, and a named group is one you can read
back and check. Three rules shape the result:

    MIN_GROUP    a category with one or two tenders is a one-off, not a
                 specialisation, and should not carry equal weight
    MAX_GROUPS   caps profile size; the largest categories win
    (the tail)   is dropped, not merged. Merging unrelated leftovers into one
                 group produces a medoid representing nothing, which is the
                 averaging problem again

Each medoid also records its share of the surviving history, so scoring can
weight a category by how often the supplier actually bids it. Nothing here
applies that weight — it is recorded, and recommend.py decides what to do
with it.

    python supplier_profile.py                 # summary across all suppliers
    python supplier_profile.py --supplier 164215 # to inspect medoids for one specific supplier
"""

import argparse
import os
from collections import namedtuple

import numpy as np
import pandas as pd

import embed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")

MIN_GROUP = 2
MAX_GROUPS = 15

Medoid = namedtuple(
    "Medoid",
    "tender title code n_tenders share centrality vector",
)


class Profile:
    """A supplier's medoids, plus what had to be discarded to get them."""

    def __init__(self, supplier, medoids, n_history, n_covered, n_categories):
        self.supplier = supplier
        self.medoids = medoids
        self.n_history = n_history          # tenders they bid on, in train
        self.n_covered = n_covered          # of those, inside a surviving group
        self.n_categories = n_categories    # distinct ОКПД2 before filtering

    @property
    def coverage(self):
        """Fraction of history the profile actually speaks for."""
        return self.n_covered / self.n_history if self.n_history else 0.0

    @property
    def matrix(self):
        """Medoid vectors as one matrix, for scoring against candidates."""
        if not self.medoids:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack([m.vector for m in self.medoids])

    @property
    def shares(self):
        return np.array([m.share for m in self.medoids], dtype=np.float32)

    def __len__(self):
        return len(self.medoids)


def pick_medoid(vectors):
    """Index of the vector most similar, on average, to the others.

    Vectors are unit length, so `vectors @ vectors.T` is the full table of
    cosine similarities. Row i averaged over the other members says how typical
    member i is of the group; the highest wins.

    Self-similarity is excluded. It is 1.0 for every row so it would not change
    which one wins, but it would inflate the centrality figure I report, and a
    reported number should stay uninflated. 
    """
    sims = vectors @ vectors.T
    n = len(vectors)
    if n == 1:
        return 0, 1.0
    centrality = (sims.sum(axis=1) - np.diag(sims)) / (n - 1)
    best = int(np.argmax(centrality))
    return best, float(centrality[best])


def build_profile(supplier, participants, train, cache,
                  min_group=MIN_GROUP, max_groups=MAX_GROUPS):
    """Medoids for one supplier, derived only from tenders they bid on."""
    lots = set(participants.loc[participants["post_num"] == supplier, "pn_lot"])
    history = train[train["pn_lot"].isin(lots)].dropna(subset=["okpd2_code"])
    if history.empty:
        return Profile(supplier, [], 0, 0, 0)

    n_history = len(history)
    n_categories = history["okpd2_code"].nunique()

    # Largest categories first, one-offs discarded, then capped.
    groups = [(code, rows) for code, rows in history.groupby("okpd2_code")
              if len(rows) >= min_group]
    groups.sort(key=lambda g: len(g[1]), reverse=True)
    groups = groups[:max_groups]

    n_covered = sum(len(rows) for _, rows in groups)
    if not n_covered:
        return Profile(supplier, [], n_history, 0, n_categories)

    medoids = []
    for code, rows in groups:
        vectors, kept = cache.rows(rows["pn_lot"].tolist())
        if len(kept) == 0:
            continue                      # nothing in this group was encoded

        best, centrality = pick_medoid(vectors)
        tender = kept[best]
        title = rows.loc[rows["pn_lot"] == tender, "purchase_name"].iloc[0]

        medoids.append(Medoid(
            tender=tender,
            title=str(title),
            code=str(code),
            n_tenders=len(rows),
            share=len(rows) / n_covered,
            centrality=centrality,
            vector=vectors[best],
        ))

    return Profile(supplier, medoids, n_history, n_covered, n_categories)


def print_profile(profile):
    """Print a profile so it can be judged by eye.

    This is the reason for medoids over an averaged vector. Fifteen readable
    Russian tender titles either look like a coherent business or they do not. 312 floats tells fairly nothing.
    """
    p = profile
    print(f"supplier {p.supplier}")
    print(f"  {p.n_history} tenders in history across {p.n_categories} categories")
    print(f"  {len(p)} medoids covering {p.n_covered} tenders "
          f"({p.coverage:.0%} of history)\n")

    if not p.medoids:
        print("  no group large enough to represent")
        return

    print(f"  {'ОКПД2':<10}{'n':>4}{'share':>8}{'centr':>8}  title")
    print("  " + "-" * 74)
    for m in p.medoids:
        title = m.title if len(m.title) <= 46 else m.title[:43] + "..."
        print(f"  {m.code:<10}{m.n_tenders:>4}{m.share:>8.0%}"
              f"{m.centrality:>8.2f}  {title}")


def load_inputs():
    participants = pd.read_csv(f"{SAMPLE_DIR}/participants.csv", low_memory=False)
    train = pd.read_csv(f"{SAMPLE_DIR}/train.csv", low_memory=False)
    return participants, train, embed.load_tenders()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--supplier", type=int, help="print one supplier's profile")
    parser.add_argument("--min-group", type=int, default=MIN_GROUP)
    parser.add_argument("--max-groups", type=int, default=MAX_GROUPS)
    args = parser.parse_args()

    participants, train, cache = load_inputs()

    if args.supplier:
        print_profile(build_profile(args.supplier, participants, train, cache,
                                    args.min_group, args.max_groups))
    else:
        suppliers = sorted(participants["post_num"].unique())
        rows = []
        for s in suppliers:
            p = build_profile(s, participants, train, cache,
                              args.min_group, args.max_groups)
            rows.append((s, p.n_history, p.n_categories, len(p), p.coverage))

        print(f"{'supplier':>10}{'history':>9}{'cats':>6}{'medoids':>9}{'coverage':>10}")
        print("-" * 44)
        for s, hist, cats, n, cov in rows:
            print(f"{s:>10}{hist:>9}{cats:>6}{n:>9}{cov:>10.0%}")

        covs = [r[4] for r in rows if r[1]]
        counts = [r[3] for r in rows if r[1]]
        print("-" * 44)
        print(f"{'mean':>10}{'':>9}{'':>6}{np.mean(counts):>9.1f}{np.mean(covs):>10.0%}")
