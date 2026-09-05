"""Sweep the semantic weight and print one table.

The weight on text similarity was picked at 0.3 without justification. This
runs a range of values through the same harness, against identical data, so
the final choice comes from a number rather than from a guess.

All rows share the same candidate pool and the same gates, so precision is
directly comparable between them. The two reference rows - random, and
structured signals with no embeddings at all - are there for scale.

    python ablate.py
    python ablate.py --repeats 30       # slower, tighter error bars
"""

import argparse
import os
import time

import pandas as pd

import embed
import evaluate
import recommend

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")

WEIGHTS = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
REPEATS = 10


def main(repeats):
    participants = pd.read_csv(f"{SAMPLE_DIR}/participants.csv", low_memory=False)
    train = pd.read_csv(f"{SAMPLE_DIR}/train.csv", low_memory=False)
    cache = embed.load_tenders()

    runs = [
        ("random", evaluate.random_ranker()),
        ("structured only", recommend.filters_ranker(participants, train)),
    ]
    for w in WEIGHTS:
        runs.append((f"  + semantic {w}",
                     recommend.filters_ranker(participants, train, cache, w)))

    print(f"{repeats} candidate orderings per configuration\n")
    print(f"{'configuration':<20}{'prec@5':>9}{'prec@10':>9}"
          f"{'prec@20':>9}{'recall@50':>11}{'sec':>7}")
    print("-" * 65)

    for label, ranker in runs:
        start = time.time()
        r = evaluate.evaluate(ranker, repeats=repeats, verbose=False)
        print(f"{label:<20}{r['precision@5']:>9.1%}{r['precision@10']:>9.1%}"
              f"{r['precision@20']:>9.1%}{r['recall@50']:>11.1%}"
              f"{time.time() - start:>7.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args()
    main(args.repeats)
