# Tender Recommendation Service

Recommends Russian public procurement tenders to a supplier, based on
what that supplier has bid on before. No questionnaires and no self-declared
categories — the profile is derived entirely from bidding behaviour.

> **Status: rewrite in progress.** The data pipeline is complete. The
> recommendation engine and evaluation harness are being rebuilt; see
> [Origin and rewrite](#origin-and-rewrite).

## Quick start

```bash
git clone https://github.com/lxnikul/Recommendation-Service.git
cd Recommendation-Service
pip install -r requirements.txt
```

The sampled dataset is committed under `sample_data/`, so there is nothing
else to set up.

## Data

`make_sample.py` is included but cannot be run without the source files. It is
here so the sample's provenance is auditable — how the 30 suppliers were
chosen, why sampling is by supplier rather than by row, and what guarantees the
subset preserves. The committed data is the output; the script is the record of
how it was produced.

The source dataset covers 44-ФЗ procurement across all sectors and regions,
supplied for a РосЭлТорг case study: tender records, line items within each
tender, and roughly 5 million supplier bids. About 2.2 GB across five files,
every one of them past GitHub's 100 MB limit.

None of it is committed. `make_sample.py` builds a 13 MB subset instead —
30 suppliers with their complete bid histories, every train-split tender
those bids reference, test candidates both inside and outside those
suppliers' categories, and all matching line items.

Sampling is by supplier, not by row. Slicing each file independently would
produce files that no longer join, and the pipeline would report zero
matches at every stage while appearing to run correctly.


## How it works

Four stages (conceptual):

**1. Offline encoding.** Every tender's text is embedded once and cached. This
is the expensive step; all later work reads the cache.

**2. Supplier profile.** Bid history is reduced to ~15 *medoids* — real past
tender titles nearest the centre of each cluster, not averaged vectors. 
Medoids are inspectable: printing them shows immediately whether the
profile is correct.

**3. Scoring.** Two hard gates — price envelope, and federal district. Beyond
that, everything is a weighted score rather than a filter:

| Signal | Method |
|---|---|
| Category | ОКПД2 prefix table with early termination |
| Region | 1.0 own region, partial credit within district |
| Semantic | best similarity against the supplier's medoids |


**4. Evaluation.** Held-out bids from the 30% split are ground truth.
precision@k and recall@k, plus ablations with individual signals disabled.

## Results

*Pending. This section will report precision@k against a filters-only baseline,
and the measured contribution of the semantic layer.*

## Origin and rewrite

This began as university coursework written with
[@fenix3f](https://github.com/fenix3f); the original is at
[fenix3f/Recomend_Service](https://github.com/fenix3f/Recomend_Service).

Running that version against the full dataset surfaced defects.

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
Cyrillic tender titles, which tokenise almost entirely to meaningless gibberish.

Structural finding mattered more than the defects.

**Filtering did nearly all the work.** On a sample supplier the funnel ran
316 - 294 - 15 candidates through category, price and region. Region alone
removed 279. With fifteen candidates remaining — all already relevant — the
ranking step could not visibly fail, which is why the broken version looked
correct.


The rewrite keeps the original architecture cheap filters ahead of expensive
semantics and changes what sits inside it:
encode once instead of per run, per-supplier medoids instead of one averaged
vector, scores instead of hard gates, and evaluation so the semantic layer's
contribution is a measured number rather than an assumption.


[MIT](LICENSE)


