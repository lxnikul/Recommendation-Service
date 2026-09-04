"""Encode tender and item text once, cache the vectors, reuse them everywhere.

A forward pass costs roughly 20 ms of CPU. With ~10,000 tenders and ~31,000
items that is minutes, and every weight change and ablation would pay it again.
The text never changes and neither does the model, so the vectors never change
either: compute once, write to disk, read thereafter.

Two caches, deliberately separate:

    cache/tenders.npz   one vector per tender, from purchase_name
    cache/items.npz     one vector per ITEM, plus the tender it belongs to

Keeping items at their own granularity is the point. Concatenating item text
onto the title and encoding that as one string dilutes the title as items pile
up, and makes the vector depend on how many items a tender happens to list.
Separate vectors keep every option open — max, mean, coverage above a
threshold — and those are decided at scoring time, not baked in here.

The encoder is rubert-tiny2: 29M parameters, a Cyrillic vocabulary, and trained
so that mean pooling produces a usable sentence vector. The bert-base-uncased
the original used has none of those properties, which is why its output on
Russian titles was noise.

Vectors are L2-normalised, so cosine similarity later is a plain dot product.

    python embed.py              # both caches
    python embed.py --tenders    # tenders only
    python embed.py --items      # items only
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "cointegrated/rubert-tiny2"
BATCH_SIZE = 64

# Resolved against this file, not the working directory, so the scripts run
# from anywhere rather than only from the repository root.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

# Titles and item names are both short. 64 tokens covers essentially all of them.
MAX_LENGTH = 64

TENDER_CACHE = os.path.join(CACHE_DIR, "tenders.npz")
ITEM_CACHE = os.path.join(CACHE_DIR, "items.npz")


class TenderCache:
    """Tender id -> one vector. A 1:1 lookup."""

    def __init__(self, ids, vectors):
        self.ids = list(ids)
        self.index = {tender: row for row, tender in enumerate(self.ids)}
        self.vectors = vectors
        self.dim = vectors.shape[1]

    def __len__(self):
        return len(self.index)

    def __contains__(self, tender):
        return tender in self.index

    def rows(self, tenders):
        """Vectors for a list of tender ids.

        Returns (matrix, kept_ids). Ids not in the cache are skipped, so the
        matrix can be shorter than the list you asked for:

            asked  = [A, B, C, D, E]    # B and D were never encoded
            matrix = [A,    C,    E]    # three rows
            kept   = [A,    C,    E]    # what those three rows are

        A row carries no label — its position is the only link back to a
        tender. So pair results with kept_ids. Pairing them with the list you
        passed in would give C's vector A's name, E's vector C's name, and so
        on down the list, and nothing would raise an error.
        """
        kept = [t for t in tenders if t in self.index]
        if not kept:
            return np.empty((0, self.dim), dtype=np.float32), []
        return self.vectors[[self.index[t] for t in kept]], kept


class ItemCache:
    """One row per item. A 1:many lookup, since a tender has several items.

    Carries each item's own ОКПД2 code alongside the vector. Those are
    full-depth (33.16.10.000) where the tender table truncates to two levels,
    so they are useful independently of anything the embeddings do.
    """

    def __init__(self, tenders, names, codes, vectors):
        self.tenders = list(tenders)
        self.names = list(names)
        self.codes = list(codes)
        self.vectors = vectors
        self.dim = vectors.shape[1]

        self.by_tender = defaultdict(list)
        for row, tender in enumerate(self.tenders):
            self.by_tender[tender].append(row)

    def __len__(self):
        return len(self.tenders)

    def __contains__(self, tender):
        return tender in self.by_tender

    def rows(self, tender):
        """Every item vector for one tender, with names and codes alongside."""
        idx = self.by_tender.get(tender, [])
        if not idx:
            return np.empty((0, self.dim), dtype=np.float32), [], []
        return (self.vectors[idx],
                [self.names[i] for i in idx],
                [self.codes[i] for i in idx])

    def rows_many(self, tenders):
        """Every item vector across several tenders — a supplier's whole history.

        Returns (matrix, parent_ids). Item counts vary wildly, from one to
        nearly two thousand per tender, so there is no way to work out from
        the matrix alone which rows came from which tender. parent_ids says,
        one entry per row.
        """
        idx = [r for t in tenders for r in self.by_tender.get(t, [])]
        if not idx:
            return np.empty((0, self.dim), dtype=np.float32), []
        return self.vectors[idx], [self.tenders[i] for i in idx]


def mean_pool(hidden, mask):
    """Average token vectors, ignoring padding positions.

    The alternative — taking [CLS], as the original did — only carries a
    sentence summary in a model fine-tuned to put one there. Untrained, [CLS]
    is just another token.
    """
    mask = mask.unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def encode(texts, device="cpu", label="", verbose=True):
    """Encode a list of strings to unit vectors.

    Duplicates are encoded once and expanded afterwards. Item names repeat
    heavily — roughly a third of them are not unique — and the model gives an
    identical answer every time, so paying for it twice is waste.
    """
    unique = sorted(set(texts))
    position = {text: i for i, text in enumerate(unique)}
    if verbose:
        print(f"  {len(texts):,} strings, {len(unique):,} unique")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    out = []
    for start in range(0, len(unique), BATCH_SIZE):
        batch = unique[start:start + BATCH_SIZE]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=MAX_LENGTH, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state
        vectors = mean_pool(hidden, enc["attention_mask"])
        vectors = torch.nn.functional.normalize(vectors, dim=1)
        out.append(vectors.cpu().numpy().astype(np.float32))

        if verbose and (start // BATCH_SIZE) % 40 == 0:
            print(f"    {label}{min(start + BATCH_SIZE, len(unique)):>7,}"
                  f" / {len(unique):,}")

    encoded = np.vstack(out)
    return encoded[[position[t] for t in texts]]


def build_tenders(device="cpu"):
    """One vector per tender, from purchase_name."""
    frames = [
        pd.read_csv(f"{SAMPLE_DIR}/{name}.csv",
                    usecols=["pn_lot", "purchase_name"], low_memory=False)
        for name in ("train", "test")
    ]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("pn_lot")
    df["purchase_name"] = df["purchase_name"].fillna("").astype(str)

    print(f"tenders: {len(df):,}")
    vectors = encode(df["purchase_name"].tolist(), device=device)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(TENDER_CACHE,
                        ids=np.array(df["pn_lot"].tolist(), dtype=object),
                        vectors=vectors)
    _report(TENDER_CACHE, vectors)
    return TenderCache(df["pn_lot"].tolist(), vectors)


def build_items(device="cpu"):
    """One vector per item, keeping the parent tender, name and ОКПД2 code."""
    df = pd.read_csv(f"{SAMPLE_DIR}/items.csv",
                     usecols=["pn_lot", "item_name", "okpd2_code"],
                     low_memory=False)
    df = df[df["item_name"].notna()].copy()
    df["item_name"] = df["item_name"].astype(str)
    df["okpd2_code"] = df["okpd2_code"].fillna("").astype(str)

    print(f"items: {len(df):,} across {df['pn_lot'].nunique():,} tenders")
    vectors = encode(df["item_name"].tolist(), device=device)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(ITEM_CACHE,
                        tenders=np.array(df["pn_lot"].tolist(), dtype=object),
                        names=np.array(df["item_name"].tolist(), dtype=object),
                        codes=np.array(df["okpd2_code"].tolist(), dtype=object),
                        vectors=vectors)
    _report(ITEM_CACHE, vectors)
    return ItemCache(df["pn_lot"], df["item_name"], df["okpd2_code"], vectors)


def _report(path, vectors):
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  wrote {path}  {vectors.shape}  {size_mb:.1f} MB\n")


def _load(path, what):
    if not os.path.exists(path):
        raise SystemExit(f"No cache at {path}. Build it with: python embed.py --{what}")
    return np.load(path, allow_pickle=True)


def load_tenders():
    data = _load(TENDER_CACHE, "tenders")
    return TenderCache(list(data["ids"]), data["vectors"])


def load_items():
    data = _load(ITEM_CACHE, "items")
    return ItemCache(list(data["tenders"]), list(data["names"]),
                     list(data["codes"]), data["vectors"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tenders", action="store_true", help="tenders only")
    parser.add_argument("--items", action="store_true", help="items only")
    parser.add_argument("--device", default="cpu",
                        help="cpu, or cuda if a GPU build of torch is installed")
    args = parser.parse_args()

    both = not (args.tenders or args.items)
    if args.tenders or both:
        build_tenders(device=args.device)
    if args.items or both:
        build_items(device=args.device)
