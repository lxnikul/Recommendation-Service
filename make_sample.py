"""Build a small, self-consistent sample of the case dataset.

The full dataset is ~2.2 GB across five files and cannot go in the repository
(GitHub rejects files over 100 MB). This writes a few-MB subset that preserves
the relationships the pipeline depends on:

  * every supplier kept has their complete bid history
  * every tender those bids point at is present — in train or in test
  * the test split contains three kinds of row, described below

A naive head(50000) of each file would break all of this and the pipeline would
report zero matches at every stage.

Run once, from the folder holding the original CSVs.

The test split is the part that matters for evaluation. It holds:

  held-out bids    tenders these suppliers actually bid on, from the 30% split.
                   This is the answer key — without it precision@k cannot be
                   computed at all.
  hard distractors same categories, not bid on. These are what a ranker has to
                   push below the real bids; filters alone cannot separate them.
  easy distractors other categories entirely. A sanity floor.

Positives alone would make precision 1.0 for any ranking. Easy distractors
alone would make the category filter look sufficient. Both are needed.
"""

import os

import pandas as pd

N_SUPPLIERS = 30        # suppliers to keep
MIN_BIDS = 20           # ...with at least this much history
MAX_BIDS = 300          # ...and not so much that they dominate the sample
N_HARD = 4000           # in-category test rows they did not bid on
N_EASY = 4000           # out-of-category test rows
PER_CHUNK = 2000        # cap taken from each chunk, so the sample spreads
CHUNK = 500_000         # rows per read; keeps memory flat on the big files
SEED = 42
OUT_DIR = "sample_data"

PARTICIPANTS = "ml_model_participants_post_num.csv"
TRAIN = "train.csv"
TEST = "test.csv"
ITEMS = "ml_model_training_data_items_2024_10_15_202412042038.csv"


def stream_keep(path, predicate):
    """Read a large CSV in chunks, concatenating only the rows we want.

    The source files are up to 800 MB; loading one whole would need several
    gigabytes of RAM once pandas has boxed the strings.
    """
    kept = []
    for chunk in pd.read_csv(path, chunksize=CHUNK, low_memory=False):
        kept.append(chunk[predicate(chunk)])
    return pd.concat(kept, ignore_index=True)


def stream_test(path, bid_lots, codes):
    """One pass over the test split, partitioned into the three groups above.

    Held-out bids are kept in full — they are the answer key and there are
    never too many. Distractors are capped, and sampled per chunk rather than
    taken from the front, so they spread across the whole file instead of
    clustering in whatever period sorts first.
    """
    held, hard, easy = [], [], []
    for chunk in pd.read_csv(path, chunksize=CHUNK, low_memory=False):
        is_bid = chunk["pn_lot"].isin(bid_lots)
        held.append(chunk[is_bid])

        rest = chunk[~is_bid]
        in_category = rest["okpd2_code"].isin(codes)
        for bucket, rows in ((hard, rest[in_category]), (easy, rest[~in_category])):
            if len(rows):
                bucket.append(rows.sample(min(len(rows), PER_CHUNK), random_state=SEED))

    def finish(parts, cap=None):
        if not parts:
            return pd.DataFrame()
        frame = pd.concat(parts, ignore_index=True)
        if cap is not None and len(frame) > cap:
            frame = frame.sample(cap, random_state=SEED)
        return frame

    return finish(held), finish(hard, N_HARD), finish(easy, N_EASY)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Choose suppliers with a useful amount of history. Sampled across the
    #    eligible range rather than taking the largest, so profile sizes vary.
    print("reading participations...")
    participants = pd.read_csv(PARTICIPANTS, low_memory=False)
    counts = participants["post_num"].value_counts()
    eligible = counts[(counts >= MIN_BIDS) & (counts <= MAX_BIDS)]
    suppliers = list(eligible.sample(
        min(N_SUPPLIERS, len(eligible)), random_state=SEED).index)
    print(f"  {len(participants):,} bids -> keeping {len(suppliers)} suppliers")

    # 2. Keep their entire bid history, not a slice of it.
    part_sample = participants[participants["post_num"].isin(suppliers)]
    bid_lots = set(part_sample["pn_lot"])
    print(f"  {len(part_sample):,} bids across {len(bid_lots):,} tenders")

    # 3. Every train row those bids reference — this is supplier history.
    print("scanning train.csv...")
    train_sample = stream_keep(TRAIN, lambda c: c["pn_lot"].isin(bid_lots))
    print(f"  {len(train_sample):,} tenders matched")

    codes = set(train_sample["okpd2_code"].dropna())
    print(f"  {len(codes)} distinct ОКПД2 values")

    # 4. Test split: answer key plus two grades of distractor.
    print("scanning test.csv...")
    held, hard, easy = stream_test(TEST, bid_lots, codes)
    test_sample = pd.concat([held, hard, easy], ignore_index=True)
    per_supplier = len(held) / max(len(suppliers), 1)
    print(f"  {len(held):,} held-out bids  ({per_supplier:.0f} per supplier)")
    print(f"  {len(hard):,} hard distractors + {len(easy):,} easy")

    if per_supplier < 10:
        print("  !! too few held-out bids per supplier — precision@10 will be noisy")

    # 5. Line items for every tender we kept, from either split.
    print("scanning items...")
    all_lots = bid_lots | set(test_sample["pn_lot"])
    items_sample = stream_keep(ITEMS, lambda c: c["pn_lot"].isin(all_lots))
    print(f"  {len(items_sample):,} line items")

    outputs = {
        "participants.csv": part_sample,
        "train.csv": train_sample,
        "test.csv": test_sample,
        "items.csv": items_sample,
    }
    print()
    for name, frame in outputs.items():
        path = os.path.join(OUT_DIR, name)
        frame.to_csv(path, index=False, encoding="utf-8")
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {path:<28} {len(frame):>8,} rows  {size_mb:>6.1f} MB")

    print(f"\nExample supplier id for testing: {suppliers[0]}")


if __name__ == "__main__":
    main()
