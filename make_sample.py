"""Build a small, self-consistent sample of the case dataset.

The full dataset is ~2.2 GB across five files and cannot go in the repository
(GitHub rejects files over 100 MB). This writes a few-MB subset into
sample_data/ that preserves the relationships the pipeline depends on:

  * every supplier kept has their complete bid history
  * every tender those bids point at is present in the train split
  * the test split holds both matching and non-matching candidates, so the
    filters have something to reject and ranking has something to rank

A naive head(50000) of each file would break all three and the pipeline would
report zero matches at every stage.

Run once, from the folder holding the original CSVs.
"""

import os

import pandas as pd

N_SUPPLIERS = 30        # suppliers to keep
MIN_BIDS = 20           # ...with at least this much history
MAX_BIDS = 300          # ...and not so much that they dominate the sample
N_POSITIVES = 6000      # test rows inside their categories
N_NEGATIVES = 6000      # test rows outside them, so filtering has work to do
PER_CHUNK = 2000        # cap taken from each chunk, so the sample spreads
CHUNK = 500_000         # rows per read; keeps memory flat on the big files
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


def stream_split(path, codes, n_pos, n_neg, seed=42):
    """One pass over a split, returning capped in- and out-of-category rows.

    Both sides are capped because "everything in these categories" can be tens
    of thousands of rows. Sampling per chunk rather than taking the first N
    keeps the result spread across the whole file instead of concentrated in
    whatever period happens to sort first.
    """
    pos, neg = [], []
    for chunk in pd.read_csv(path, chunksize=CHUNK, low_memory=False):
        mask = chunk["okpd2_code"].isin(codes)
        for side, rows in ((pos, chunk[mask]), (neg, chunk[~mask])):
            if len(rows):
                side.append(rows.sample(min(len(rows), PER_CHUNK), random_state=seed))

    def finish(parts, cap):
        if not parts:
            return pd.DataFrame()
        frame = pd.concat(parts, ignore_index=True)
        return frame.sample(min(len(frame), cap), random_state=seed)

    return finish(pos, n_pos), finish(neg, n_neg)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Choose suppliers with a useful amount of history. Too few bids and
    #    there is nothing to build a profile from; too many and one supplier
    #    swamps the sample.
    print("reading participations...")
    participants = pd.read_csv(PARTICIPANTS, low_memory=False)
    counts = participants["post_num"].value_counts()
    eligible = counts[(counts >= MIN_BIDS) & (counts <= MAX_BIDS)]
    suppliers = list(eligible.head(N_SUPPLIERS).index)
    print(f"  {len(participants):,} bids -> keeping {len(suppliers)} suppliers")

    # 2. Keep their entire bid history, not a slice of it.
    part_sample = participants[participants["post_num"].isin(suppliers)]
    lots = set(part_sample["pn_lot"])
    print(f"  {len(part_sample):,} bids across {len(lots):,} tenders")

    # 3. Every train row those bids reference.
    print("scanning train.csv...")
    train_sample = stream_keep(TRAIN, lambda c: c["pn_lot"].isin(lots))
    print(f"  {len(train_sample):,} tenders matched")

    # 4. The categories those suppliers actually work in.
    codes = set(train_sample["okpd2_code"].dropna())
    print(f"  {len(codes)} distinct ОКПД2 values")

    # 5. Test candidates: everything in those categories, plus a block of
    #    rows outside them so the ОКПД2 filter has real work to do.
    print("scanning test.csv...")
    positives, negatives = stream_split(TEST, codes, N_POSITIVES, N_NEGATIVES)
    test_sample = pd.concat([positives, negatives], ignore_index=True)
    print(f"  {len(positives):,} in-category + {len(negatives):,} out-of-category")
    if negatives.empty:
        print("  !! no out-of-category rows — the ОКПД2 filter cannot be tested")

    # 6. Line items for every tender we kept, from either split. These carry
    #    full-depth ОКПД2 codes and far more specific text than tender titles.
    print("scanning items...")
    all_lots = lots | set(test_sample["pn_lot"])
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

    print(f"\nExample supplier id for main.py: {suppliers[0]}")


if __name__ == "__main__":
    main()
