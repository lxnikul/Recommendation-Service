"""Rank tenders for a supplier using structured signals only.

No embeddings, no machine learning. Price and federal district are hard gates;
category and region are weighted scores. This is the baseline the semantic
layer has to beat — if it cannot, the embeddings are not earning their compute.

Run directly to score it:

    python recommend.py
"""

import os

import pandas as pd

# Resolved against this file, not the working directory, so the scripts run
# from anywhere rather than only from the repository root.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")

# Weights for the scored signals. They sum to 1 so the total reads as 0..1.
W_CATEGORY = 0.6
W_REGION = 0.4

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


def score_candidates(profile, candidates):
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

    df["score"] = W_CATEGORY * df["category_score"] + W_REGION * df["region_score"]
    return df.sort_values("score", ascending=False)


def filters_ranker(participants, train):
    """Build a ranking function of the shape evaluate.py expects."""
    profiles = {}

    def rank(supplier, candidates):
        if supplier not in profiles:
            profiles[supplier] = build_profile(supplier, participants, train)
        profile = profiles[supplier]
        if profile is None:
            return []
        return score_candidates(profile, candidates)["pn_lot"].tolist()

    return rank


if __name__ == "__main__":
    import argparse

    import evaluate

    parser = argparse.ArgumentParser(description="Score the filters-only ranker.")
    parser.add_argument("--repeats", type=int, default=evaluate.DEFAULT_REPEATS,
                        help="candidate orderings to average over")
    args = parser.parse_args()

    participants = pd.read_csv(f"{SAMPLE_DIR}/participants.csv", low_memory=False)
    train = pd.read_csv(f"{SAMPLE_DIR}/train.csv", low_memory=False)

    print("Filters only: price and district gates, category and region scores\n")
    evaluate.evaluate(filters_ranker(participants, train), repeats=args.repeats)
