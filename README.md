# Tender Recommendation Service

Recommends Russian public procurement tenders to a supplier, based on
what that supplier has bid on before. No questionnaires and no self-declared
categories — the profile is derived entirely from bidding behaviour.

**precision@10 = 74.8%** against a 0.3% random baseline, on held-out bids.
Structured signals alone reach 43.2%; text similarity accounts for the rest.

## Quick start

```bash
git clone https://github.com/lxnikul/Recommendation-Service.git
cd Recommendation-Service
pip install -r requirements.txt
python embed.py          # downloads the encoder (~120 MB), 2-3 min
python recommend.py
```

The sampled dataset is committed under `sample_data/`, so there is nothing
else to set up. `embed.py` builds the vector cache, which is derived data and
is not committed.

## Data

The source dataset covers 44-ФЗ procurement across all sectors and regions,
supplied for a РосЭлТорг case study: tender records, line items within each
tender, and roughly 5 million supplier bids. About 2.2 GB across five files,
every one of them past GitHub's 100 MB limit.

None of it is committed. `make_sample.py` builds a 7 MB subset instead —
30 suppliers with their complete bid histories, every train-split tender
those bids reference, test candidates both inside and outside those
suppliers' categories, and all matching line items. That comes to 2,107 bids,
1,463 history tenders, 8,643 candidates and 31,610 line items.

Sampling is by supplier, not by row. Slicing each file independently would
produce files that no longer join, and the pipeline would report zero
matches at every stage while appearing to run correctly.

The test split holds three kinds of row: tenders the sampled suppliers
actually bid on, which are the answer key; tenders in their categories that
they did not bid on; and tenders outside their categories. All three are
needed. Answers alone make precision 1.0 for any ranking, and easy
distractors alone would make category filtering look sufficient.

`make_sample.py` cannot be run without the source files. It is here so the
sample's provenance is auditable — how the suppliers were chosen, and what
the subset is guaranteed to preserve.

## How it works

**1. Offline encoding.** Every tender's text is embedded once and cached with
rubert-tiny2, mean-pooled and L2-normalised. This is the expensive step; all
later work reads the cache.

**2. Supplier profile.** Bid history is reduced to medoids — for each ОКПД2
category the supplier works in, the one real past tender most typical of that
group. An averaged vector cannot be inspected. Titles can be.

Categories with fewer than two tenders are dropped as one-offs, the largest
fifteen are kept, and the tail is dropped rather than merged; pooling
unrelated leftovers produces a medoid representing nothing. In practice this
gives 3.4 medoids per supplier covering 96% of their history.

**3. Scoring.** Two hard gates — price envelope, and federal district. Beyond
that, everything is a weighted score rather than a filter:

| Signal | Method |
|---|---|
| Category | ОКПД2 prefix table with early termination |
| Region | 1.0 own region, partial credit within district |
| Semantic | best similarity against the supplier's medoids |

Maximum rather than mean, so a candidate is compared against the closest
thing the supplier does rather than an average of everything they do.

**4. Evaluation.** Held-out bids from the 30% split are the ground truth.
`evaluate.py` takes a ranking function rather than data, so the same code
judges every version, and averages over shuffled candidate orderings.

## Results

19 of 30 suppliers have enough held-out bids to score (10 or more). Every
figure is averaged over 30 candidate orderings.

| Configuration | precision@10 | recall@50 |
|---|---|---|
| Random ranking | 0.3% | 0.5% |
| Structured signals only | 43.2% | 56.3% |
| **+ semantic similarity** | **74.8%** | **68.8%** |

Structured signals alone — ОКПД2 prefix depth and region, gated by price and
federal district — reach 43% with no machine learning of any kind. Text
similarity takes it to 75%.

The semantic weight barely matters. Across a tenfold range from 0.1 to 1.0,
precision@10 moves by 0.6 points and precision@5 is identical from 0.1 to
0.7. The whole gain arrives at the smallest weight tested, which places the
text signal precisely: the structured score takes six discrete values, and
the embeddings order candidates within those groups rather than moving them
between. The shipped weight is 0.1, the lowest setting that captures it.

### How much of this is repeat procurement

22% of held-out bids have a title that appears verbatim in the supplier's
history. By vector similarity, 35% exceed 0.98 and 64% exceed 0.90, with a
median of 0.939. Public procurement repeats constantly and finding a
returning tender is the most useful thing this system does, but it means the
headline figure is measured on a pool where a third of the wins are the same
lot coming round again.

So the obvious question is how the system does once those are taken away.
`analyse_novel.py` removes candidates that closely match the supplier's
history from the pool entirely — filtered on history text and candidate text
only, never on labels, so correct answers and distractors are treated
identically and nothing leaks.

| Pool | candidates | answer key | precision@10 | recall@50 |
|---|---|---|---|---|
| All candidates | 8,643 | 30 | 73.4% | 68.5% |
| Matches above 0.98 removed | 8,628 | 19 | 62.2% | 68.2% |
| Matches above 0.90 removed | 8,586 | 11 | 27.8% | 63.4% |

Precision@10 appears to collapse. In fact, it does not.

**Precision divides by a constant.** `precision@10 = (correct in top 10) / 10`.
Remove correct answers from the pool and the numerator falls while the
denominator stays pinned at ten.

**Recall divides by the answer key.** `recall@50 = (correct in top 50) / (total
correct)`. Remove correct answers and both parts fall together, so the ratio
holds.

That is the entire difference. Working it through with the numbers above:

```
full pool          0.734 x 10 = 7.3 correct in the top 10, out of 30
                                recall@10 = 7.3 / 30 = 24%

repeats removed    0.278 x 10 = 2.8 correct in the top 10, out of 11
                                recall@10 = 2.8 / 11 = 25%
```

**The same proportion of correct answers reaches the top ten either way.**
Precision differs only because there are fewer of them in existence.

The same conclusion from the other direction: the answer key thins from 30 in
8,643 (a 0.35% base rate) to 11 in 8,586 (0.13%), a factor of 2.7. Precision
falls by a factor of 2.6. Dividing one by the other, lift over random is
**211×** on the full pool and **217×** on the reduced one — unchanged.

Ranking quality on unfamiliar tenders matches that on familiar ones. There are
simply fewer of them to find, and every method — this one, a perfect one, a
random one — scores lower in absolute precision when needles get rarer.

### Similarity to history as a predictor on its own

The filter used above is itself a classifier, and it is worth reporting
separately. Per supplier, at threshold 0.98 it removed 15 candidates of which
11 were tenders the supplier bid on. At 0.90 it removed 57, of which 19 were.

| Threshold | Precision | Recall |
|---|---|---|
| >0.98 | 73% (11 of 15) | 37% (11 of 30) |
| >0.90 | 33% (19 of 57) | 63% (19 of 30) |

A candidate that closely resembles something a supplier has already bid on is
bid on again roughly three times in four, and that rule needs no ranking model
at all. Loosening the threshold trades that precision for coverage in the
usual way.

## Origin and rewrite

This began as university coursework written with
[@fenix3f](https://github.com/fenix3f); the original is at
[fenix3f/Recomend_Service](https://github.com/fenix3f/Recomend_Service).

Running that version against the full dataset surfaced several defects.

**Ranking used two incompatible index spaces.** `find_similar_tenders` built
its vector array from individual keyword-matched *words*, then used the
resulting `argsort` indices to slice the *row*-level candidate table. A title
with three matching words contributed three entries; one with none contributed
zero. Returned tenders therefore bore no relation to their own similarity
scores — four of ten results in a sample run contained no matched word at all.
With more matched words than candidate rows it raises `IndexError` instead.

**The supplier profile was a leaked loop variable.** `main_start` correctly
built a dict of per-category vectors, then iterated it to print progress and
returned the loop variable afterwards. Thirteen of fourteen categories were
computed, printed and discarded; ranking ran against whichever one happened to
be last in insertion order.

**The encoder had no Russian vocabulary.** `bert-base-uncased` was applied to
Cyrillic tender titles, which tokenise almost entirely to meaningless
fragments.

One structural finding mattered more than the defects.

**Filtering did nearly all the work.** On a sample supplier the funnel ran
316 → 294 → 15 candidates through category, price and region. Region alone
removed 279. With fifteen candidates remaining — all already relevant — the
ranking step could not visibly fail, which is why the broken version looked
correct.

The rewrite keeps the original architecture — cheap filters ahead of
expensive semantics — and changes what sits inside it: encode once instead of
per run, per-supplier medoids instead of one averaged vector, scores instead
of hard gates, and evaluation so the semantic layer's contribution is a
measured number rather than an assumption.

## Known limitations

- **The split is random over tenders, not chronological.** A tender and a
  near-duplicate of it can land on opposite sides, so the figures are likely
  optimistic for deployment. Splitting on `min_publish_date` would fix it and
  invalidate every number above.
- **19 suppliers, precision@10 ranging from 5% to 100%.** Differences smaller
  than a few points are below the resolution of this fixture.
- **`okpd2_code` is not always a code.** It also holds labels from another
  taxonomy — `atom`, `com`, `drug` — in 11 of the 30 suppliers. Grouping still
  works, but the logic behind those labels is undocumented.
- **Some titles are procedural boilerplate** («Аукцион в электронной форме на
  право заключения…») and carry no product information. They attract the
  highest centrality precisely because they are formulaic. Filtering them out
  is worth exploring.
- **One product appears under several ОКПД2 codes.** One supplier has
  byte-identical medoid titles under 27.4 and 32.5, both defensible for a UV
  lamp. Scoring is unaffected, since it takes the maximum across medoids, but
  slots are: a supplier at the fifteen-medoid cap spent three of them on the
  same product.
- **Line items are sampled and encoded but not used.** They carry full-depth
  ОКПД2 codes (`33.16.10.000`) where the tender table truncates to two levels
  (`10.1`), which would give the category score far more resolution.
- Recommendations are generated per supplier on demand; there is no ranking
  model trained across suppliers.

## Licence

[MIT](LICENSE)
