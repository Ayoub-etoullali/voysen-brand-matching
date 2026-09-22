# Voysen — Brand Association Case

Reconciles the retailer brand referential (`retailer_brands.csv`, 280,960 rows
from 78 marketplaces) against the Voysen brand referential
(`voysen_brands.csv`, 12,153 brands), producing one association per retailer
row: `(id_brand_voysen, brand_voysen, url_source, source)`, and creating new
Voysen brands automatically when no real match exists.

## Run it

```bash
cd src
pip install pandas numpy scikit-learn scipy
python match_brands.py
```

Runs in ~90 seconds on the full 280,960-row dataset on a single core, no
GPU, no external services. Outputs land in `../output/`.

## Approach

Three fully automated stages (see `src/match_brands.py` docstring and inline
comments for the detailed reasoning):

1. **Normalize + exact match** (`src/normalize.py`) — fold accents/case,
   strip punctuation and legal-entity suffixes (Inc, GmbH, SARL, ...), then
   do a direct dictionary lookup against Voysen's normalized brand names.
2. **TF-IDF character n-gram + cosine similarity** for everything the exact
   stage missed. A compiled fuzzy-matching library (rapidfuzz) wasn't
   installable in this offline environment, so the matching is vectorized
   instead: `scikit-learn`'s `TfidfVectorizer(analyzer="char_wb")` +
   batched sparse cosine similarity. This scales to 280k x 12k comparisons
   without ever materializing a dense similarity matrix, and it's
   language-agnostic (works the same on Latin, Cyrillic, Japanese, Arabic
   brand names).
3. **New brand creation** for anything still unmatched. Unmatched names are
   optionally self-clustered (so 3 spelling variants of one unmatched brand
   become 1 new brand, not 3) before ids are assigned. A data-quality guard
   flags candidates that look like full product titles rather than brand
   names (`> 6 words` or `> 60 chars`) — they still get an id so no row is
   dropped, but they're routed to `non_brand_review_queue.csv` instead of
   silently joining the trusted brand referential.

Every threshold is a named constant at the top of `match_brands.py`:

| Constant | Value | Meaning |
|---|---|---|
| `SIM_MATCH_THRESHOLD` | 0.83 | cosine sim ≥ this → auto-accept as an existing Voysen brand |
| `SIM_REVIEW_THRESHOLD` | 0.70 | [0.70, 0.83) → still auto-matched, but flagged for optional QA |
| `SIM_NEW_BRAND_CLUSTER_THRESHOLD` | 0.90 | merge near-duplicate unmatched names into one new brand |

They were set by inspecting the similarity-score distribution and
spot-checking samples near each boundary — documented as a **known
limitation**: with a small labeled sample (~500 pairs), precision/recall
could be measured and the thresholds tuned properly instead of by eye.

## Results on the full dataset

| Outcome | Rows | Share |
|---|---:|---:|
| Exact match | 1,564 | 0.6% |
| Fuzzy match — high confidence | 8,643 | 3.1% |
| Fuzzy match — review band | 31,264 | 11.1% |
| New brand created | 228,439 | 81.3% |
| Unlabeled (no usable brand text) | 11,050 | 3.9% |

185,478 new Voysen brands were created (many retailer rows collapse onto the
same new brand). 5,152 new-brand candidates were flagged as likely non-brand
text. Full numbers: `output/match_report.json`.

The high share of "new brand" is expected, not a matching failure: Voysen's
referential is a curated, industry-specific (beauty/wellness) brand list,
while the retailer catalogs are broad, general marketplaces — the two
universes only partially overlap by design (e.g. no exact "Nike" or "Adidas"
entry exists in the Voysen list at all).

## Deliverables

- `src/normalize.py`, `src/match_brands.py` — the pipeline (documented, no
  notebook required to read it)
- `output/brand_associations.csv` — **the required deliverable**: one row
  per retailer row with `id_brand_voysen, brand_voysen, url_source, source`,
  plus `match_type` / `match_score` / `likely_non_brand` for auditability
- `output/voysen_brands_augmented.csv` — original 12,153 Voysen brands +
  185,478 newly created brands
- `output/review_queue.csv` — medium-confidence matches, for optional human
  spot-checking (not required to run the pipeline)
- `output/non_brand_review_queue.csv` — new-brand candidates that look like
  product titles rather than brand names
- `output/match_report.json` — run metrics, thresholds, timing
- `Voysen_Brand_Matching_Presentation.pptx` — the restitution deck

## Known limitations & next steps

See the "Limits & next steps" slide in the presentation, and the comments in
`match_brands.py`. In short: no compiled fuzzy-match library offline (TF-IDF
n-grams is a solid substitute but not identical to edit-distance matching);
near-duplicate clustering of new brands is memory-guarded above 40k unique
unmatched names in this environment; thresholds need a labeled sample to be
rigorously validated rather than eyeballed.
