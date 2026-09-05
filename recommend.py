"""Rank tenders for a supplier.

Price and federal district are hard gates. Category, region and — optionally —
text similarity are weighted scores.

The semantic signal is off by default. `--semantic 0` reproduces the
structured-only baseline exactly, so any difference between runs is
attributable to the embeddings rather than to anything else moving.

    python recommend.py                  # structured signals only
    python recommend.py --semantic 0.3   # plus medoid similarity
"""

import os

import numpy as np
import pandas as pd

# Resolved against this file, not the working directory, so the scripts run
# from anywhere rather than only from the repository root.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")

# Weights for the scored signals. The total is divided by their sum, so setting
# W_SEMANTIC to 0 leaves the category:region ratio exactly as the baseline had
# it — the comparison is then clean.
W_CATEGORY = 0.6
W_REGION = 0.4
W_SEMANTIC = 0.0

# Region scoring.
S_SAME_REGION = 1.0
S_SAME_DISTRICT = 0.6

# Category scoring by how many ОКПД2 segments agree. Codes in this dataset are
# truncated to two levels in the tender table (10.1, 20.2) and carry four in
# the items table (33.16.10.000), so all four depths are reachable.
CATEGORY_SCORE = {1: 0.3, 2: 0.6, 3: 0.8, 4: 1.0}

# Price envelope, relative to what the supplier has bid on before. Generous on
# purpose — a hard gate that is too tight collapses the candidate pool to
# nothing and leaves ranking with no work to do.
PRICE_FLOOR_FACTOR = 0.5
PRICE_CEILING_FACTOR = 3.0

# Russian federal districts by region code. Buryatia (03) and Zabaykalsky (75)
# moved from Siberian to Far Eastern in 2018; this reflects the post-2018
# arrangement, which matches the 2021-2024 data.
REGION_TO_DISTRICT = {}
for _district, _codes in {
    "ЦФО": [31, 32, 33, 36, 37, 40, 44, 46, 48, 50, 57, 62, 67, 68, 69, 71, 76, 77],
    "СЗФО": [10, 11, 29, 35, 39, 47, 51, 53, 60, 78, 83],
    "ЮФО": [1, 8, 23, 30, 34, 61, 91, 92],
    "СКФО": [5, 6, 7, 9, 15, 20, 26],
    "ПФО": [2, 12, 13, 16, 18, 21, 43, 52, 56, 58, 59, 63, 64, 73],
    "УФО": [45, 66, 72, 74, 86, 89],
    "СФО": [4, 17, 19, 22, 24, 38, 42, 54, 55, 70],
    "ДФО": [3, 14, 25, 27, 28, 41, 49, 65, 75, 79, 87],
}.items():
    for _code in _codes:
        REGION_TO_DISTRICT[_code] = _district


def build_category_lookup(codes):
    """Expand a supplier's ОКПД2 codes into a prefix table.

    Every prefix of every code becomes a key, mapped to the score a candidate
    earns by agreeing that far. Because all prefixes are expanded, the table is
    prefix-closed: if a prefix is absent, no longer prefix can be present.
    That property is what lets scoring stop at the first miss.
    """
    lookup = {}
    for code in codes:
        segments = str(code).split(".")
        key = ""
        for depth, segment in enumerate(segments, start=1):
            key = f"{key}.{segment}" if key else segment
            score = CATEGORY_SCORE.get(depth, 1.0)
            if score > lookup.get(key, 0.0):
                lookup[key] = score
    return lookup


def category_score(code, lookup):
    """Best score any prefix of `code` achieves against the lookup.

    Walks shortest prefix to longest. Scores rise with depth and hits are
    contiguous, so the last hit before the first miss is the maximum — no need
    to check the remaining prefixes.
    """
    best, key = 0.0, ""
    for segment in str(code).split("."):
        key = f"{key}.{segment}" if key else segment
        if key not in lookup:
            break
        best = lookup[key]
    return best


def build_profile(supplier, participants, train):
    """Derive everything the ranker knows about a supplier from their bids.

    Nothing here is self-declared: categories, regions and price range all come
    from tenders they actually bid on in the train split.
    """
    lots = set(participants.loc[participants["post_num"] == supplier, "pn_lot"])
    history = train[train["pn_lot"].isin(lots)]
    if history.empty:
        return None

    prices = history["lot_price"].dropna()
    prices = prices[prices > 0]  # zero-price lots carry no envelope information

    regions = set(history["region_code"].dropna().astype(int))
    districts = {REGION_TO_DISTRICT.get(r) for r in regions}
    districts.discard(None)

    return {
        "categories": build_category_lookup(history["okpd2_code"].dropna().unique()),
        "regions": regions,
        "districts": districts,
        "price_min": prices.min() * PRICE_FLOOR_FACTOR if len(prices) else 0.0,
        "price_max": prices.max() * PRICE_CEILING_FACTOR if len(prices) else float("inf"),
        "n_history": len(history),
    }


def semantic_score(candidates, medoids, cache):
    """Each candidate's best similarity to any of the supplier's medoids.

    `medoids` is a (k x 312) matrix of unit vectors, `candidates` a frame of
    tenders. One matrix multiply gives every candidate-against-every-medoid
    similarity at once, and the row maximum is the score: how close is this
    tender to the nearest thing the supplier actually does.

    Maximum, not mean. A supplier who does dairy and fish should score a fish
    lot on its resemblance to their fish work, not on its average resemblance
    to dairy and fish together — that average is the problem medoids exist to
    avoid, reintroduced at scoring time.

    Returns a Series indexed by tender id, so the caller joins by id rather
    than by position. Any candidate missing from the cache gets 0.
    """
    vectors, kept = cache.rows(candidates["pn_lot"].tolist())
    if len(kept) == 0 or medoids.size == 0:
        return pd.Series(0.0, index=candidates["pn_lot"])

    best = (vectors @ medoids.T).max(axis=1)
    return pd.Series(best, index=kept)


def score_candidates(profile, candidates, medoids=None, cache=None,
                     w_semantic=W_SEMANTIC):
    """Apply the gates, then score whatever survives. Returns a sorted frame."""
    df = candidates

    # Hard gates. Rows failing either are removed, not penalised.
    price = df["lot_price"].fillna(0)
    in_price = (price >= profile["price_min"]) & (price <= profile["price_max"])

    district = df["region_code"].map(REGION_TO_DISTRICT)
    in_district = district.isin(profile["districts"])

    df = df[in_price & in_district].copy()
    if df.empty:
        return df

    # Scored signals.
    df["category_score"] = df["okpd2_code"].map(
        lambda c: category_score(c, profile["categories"]))

    own_region = df["region_code"].isin(profile["regions"])
    df["region_score"] = own_region.map({True: S_SAME_REGION, False: S_SAME_DISTRICT})

    total = W_CATEGORY * df["category_score"] + W_REGION * df["region_score"]
    weight = W_CATEGORY + W_REGION

    if w_semantic > 0 and cache is not None and medoids is not None:
        sims = semantic_score(df, medoids, cache)
        df["semantic_score"] = df["pn_lot"].map(sims).fillna(0.0)
        total = total + w_semantic * df["semantic_score"]
        weight += w_semantic

    # Dividing by the weight sum keeps the score on 0..1 and, more usefully,
    # leaves the category:region ratio untouched when w_semantic is 0.
    df["score"] = total / weight
    return df.sort_values("score", ascending=False)


def filters_ranker(participants, train, cache=None, w_semantic=W_SEMANTIC):
    """Build a ranking function of the shape evaluate.py expects.

    With w_semantic at 0 no cache is touched and no embedding is read, so the
    structured baseline runs on a machine that has never built one.
    """
    import supplier_profile

    profiles, medoid_matrices = {}, {}

    def rank(supplier, candidates):
        if supplier not in profiles:
            profiles[supplier] = build_profile(supplier, participants, train)
            if w_semantic > 0 and cache is not None:
                prof = supplier_profile.build_profile(
                    supplier, participants, train, cache)
                medoid_matrices[supplier] = prof.matrix
            else:
                medoid_matrices[supplier] = None

        profile = profiles[supplier]
        if profile is None:
            return []
        scored = score_candidates(profile, candidates,
                                  medoid_matrices[supplier], cache, w_semantic)
        return scored["pn_lot"].tolist()

    return rank


if __name__ == "__main__":
    import argparse

    import evaluate

    parser = argparse.ArgumentParser(description="Score the ranker.")
    parser.add_argument("--repeats", type=int, default=evaluate.DEFAULT_REPEATS,
                        help="candidate orderings to average over")
    parser.add_argument("--semantic", type=float, default=W_SEMANTIC,
                        help="weight on medoid similarity; 0 disables it")
    args = parser.parse_args()

    participants = pd.read_csv(f"{SAMPLE_DIR}/participants.csv", low_memory=False)
    train = pd.read_csv(f"{SAMPLE_DIR}/train.csv", low_memory=False)

    cache = None
    if args.semantic > 0:
        import embed
        cache = embed.load_tenders()
        print(f"Structured signals + medoid similarity (weight {args.semantic})\n")
    else:
        print("Structured signals only: category and region, no embeddings\n")

    evaluate.evaluate(
        filters_ranker(participants, train, cache, args.semantic),
        repeats=args.repeats,
    )
