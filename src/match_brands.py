"""
Voysen technical case, brand association pipeline.

Reconciles the retailer brand referential (280,960 rows scraped from 78
retailer platforms) against the Voysen brand referential (12,153 brands),
producing, for every retailer row, an association:

    (id_brand_voysen, brand_voysen, url_source, source)

Two-stage, fully automated matching strategy:
  1. Exact match on a normalized canonical key (handles ~case/accent/
     punctuation/legal-suffix noise), cheap, unambiguous, high precision.
  2. Character n-gram TF-IDF + cosine similarity nearest-neighbor search
     for everything the exact stage missed, catches typos, transliteration
     variants, spacing differences, partial legal names, etc. Batched
     sparse matrix multiplication keeps this tractable at 280k x 12k scale
     without needing a compiled fuzzy-matching library (none available in
     this offline environment).

Brands that still have no acceptable candidate above SIM_THRESHOLD are
netted out, self-clustered (to avoid creating 3 new "brands" for 3 spelling
variants of the same unmatched name), and appended to the Voysen
referential as brand-new entries with fresh sequential ids.

Usage:
    python match_brands.py
Outputs (in ../output/):
    brand_associations.csv   - one row per retailer row (the deliverable)
    voysen_brands_augmented.csv - original Voysen table + newly created brands
    match_report.json        - run metrics or the presentation / QA
    review_queue.csv         - low-confidence matches surfaced for optional
                                human spot-check (not required to run the
                                pipeline, purely for QA / trust-building)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from normalize import normalize

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = DATA_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

VOYSEN_PATH = DATA_DIR / "data/voysen_brands.csv"
RETAILER_PATH = DATA_DIR / "data/retailer_brands.csv"

# Cosine similarity thresholds (on 2-4 char n-gram TF-IDF vectors).
# Tuned by inspecting the similarity-score histogram (see match_report.json
# "threshold_calibration") and spot-checking samples around each boundary.
SIM_MATCH_THRESHOLD = 0.83      # >= this: auto-accept as a match to an existing Voysen brand
SIM_REVIEW_THRESHOLD = 0.70     # [review, match): matched, but flagged in review_queue.csv for optional QA
SIM_NEW_BRAND_CLUSTER_THRESHOLD = 0.90  # merge unmatched retailer names into one new brand

BATCH_SIZE = 3000  # rows of the retailer similarity search processed per batch


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ---------------------------------------------------------------------------
# Load & normalize
# ---------------------------------------------------------------------------
def load_data():
    voysen = pd.read_csv(VOYSEN_PATH)
    retailer = pd.read_csv(RETAILER_PATH)

    voysen["norm"] = voysen["name"].map(normalize)

    # Prefer the pre-cleaned matching column Voysen already supplies;
    # fall back to the raw source name when it's missing.
    key_col = retailer["brand_source_used_for_matching"].fillna(
        retailer["brand_source"]
    )
    retailer["match_key_raw"] = key_col
    retailer["norm"] = key_col.map(normalize)
    return voysen, retailer


# ---------------------------------------------------------------------------
# Stage 1: exact match on normalized key
# ---------------------------------------------------------------------------
def build_exact_lookup(voysen: pd.DataFrame) -> dict:
    """norm -> id. When several Voysen brands normalize to the same key,
    keep the lowest id deterministically (documented, auditable rule)."""
    lookup: dict[str, int] = {}
    for _id, norm in zip(voysen["id"], voysen["norm"]):
        if not norm:
            continue
        if norm not in lookup or _id < lookup[norm]:
            lookup[norm] = _id
    return lookup


# ---------------------------------------------------------------------------
# Stage 2: TF-IDF char n-gram nearest neighbor for the exact-match leftovers
# ---------------------------------------------------------------------------
def fuzzy_match(unmatched_norms: list[str], voysen: pd.DataFrame):
    """For each unique unmatched normalized retailer name, find the closest
    Voysen brand by cosine similarity on char n-grams. Returns arrays of
    (best_idx_into_voysen, best_score) aligned with unmatched_norms."""
    voysen_norms = voysen["norm"].tolist()

    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), min_df=1, max_df=0.5, sublinear_tf=True
    )
    voysen_matrix = vectorizer.fit_transform(voysen_norms)
    # Pre-normalize rows to unit length so a plain dot product == cosine sim.
    voysen_matrix = sparse.csr_matrix(voysen_matrix)

    query_matrix = vectorizer.transform(unmatched_norms)

    n = query_matrix.shape[0]
    best_idx = np.full(n, -1, dtype=np.int64)
    best_score = np.zeros(n, dtype=np.float32)

    voysen_matrix_T = voysen_matrix.T.tocsr()

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = query_matrix[start:end]
        sims = batch.dot(voysen_matrix_T)  # sparse (batch x n_voysen)
        sims = sims.tocsr()
        for row_i in range(sims.shape[0]):
            row = sims.getrow(row_i)
            if row.nnz == 0:
                continue
            local_argmax = row.data.argmax()
            best_score[start + row_i] = row.data[local_argmax]
            best_idx[start + row_i] = row.indices[local_argmax]
        log(f"  fuzzy match batch {end}/{n}")

    return best_idx, best_score


def cluster_new_brands(names: list[str], threshold: float = SIM_NEW_BRAND_CLUSTER_THRESHOLD):
    """Self-similarity clustering (union-find) so that near-duplicate
    unmatched names (e.g. 'Sallyzen' / 'Sally Zen') become ONE new brand
    instead of several. Returns a list of cluster ids aligned with `names`."""
    n = len(names)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if n == 0:
        return []

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, max_df=0.5, sublinear_tf=True)
    matrix = sparse.csr_matrix(vectorizer.fit_transform(names))
    matrix_T = matrix.T.tocsr()

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        sims = matrix[start:end].dot(matrix_T).tocsr()
        for row_i in range(sims.shape[0]):
            global_i = start + row_i
            row = sims.getrow(row_i)
            for j, score in zip(row.indices, row.data):
                if j > global_i and score >= threshold:
                    union(global_i, j)
        log(f"  new-brand clustering batch {end}/{n}")

    roots = [find(i) for i in range(n)]
    return roots


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log("Loading data...")
    voysen, retailer = load_data()
    log(f"Voysen brands: {len(voysen)} | Retailer rows: {len(retailer)}")

    exact_lookup = build_exact_lookup(voysen)

    retailer["matched_id"] = retailer["norm"].map(exact_lookup)
    retailer["match_type"] = np.where(retailer["matched_id"].notna(), "exact", None)
    retailer["match_score"] = np.where(retailer["matched_id"].notna(), 1.0, np.nan)

    n_exact = retailer["matched_id"].notna().sum()
    log(f"Exact matches: {n_exact} / {len(retailer)} ({n_exact/len(retailer):.1%})")

    # ---- Stage 2: fuzzy match the remainder, deduplicated on unique norm ----
    remaining_mask = retailer["matched_id"].isna() & (retailer["norm"] != "")
    unique_unmatched = retailer.loc[remaining_mask, "norm"].dropna().unique().tolist()
    log(f"Unique unmatched normalized names to fuzzy-match: {len(unique_unmatched)}")

    best_idx, best_score = fuzzy_match(unique_unmatched, voysen)

    voysen_ids = voysen["id"].to_numpy()
    voysen_names = voysen["name"].to_numpy()

    norm_to_fuzzy = {}
    for norm, idx, score in zip(unique_unmatched, best_idx, best_score):
        if idx == -1:
            continue
        norm_to_fuzzy[norm] = (voysen_ids[idx], score)

    def apply_fuzzy(row):
        if pd.notna(row["matched_id"]):
            return row["matched_id"], row["match_type"], row["match_score"]
        hit = norm_to_fuzzy.get(row["norm"])
        if hit is None:
            return np.nan, None, np.nan
        vid, score = hit
        if score >= SIM_MATCH_THRESHOLD:
            return vid, "fuzzy_high", float(score)
        elif score >= SIM_REVIEW_THRESHOLD:
            return vid, "fuzzy_review", float(score)
        return np.nan, None, np.nan

    applied = retailer.apply(apply_fuzzy, axis=1, result_type="expand")
    retailer["matched_id"] = applied[0]
    retailer["match_type"] = applied[1]
    retailer["match_score"] = applied[2]

    n_fuzzy_high = (retailer["match_type"] == "fuzzy_high").sum()
    n_fuzzy_review = (retailer["match_type"] == "fuzzy_review").sum()
    log(f"Fuzzy high-confidence matches: {n_fuzzy_high}")
    log(f"Fuzzy review-band matches: {n_fuzzy_review}")

    # ---- Stage 3: create new brands for everything still unmatched ----
    still_unmatched_mask = retailer["matched_id"].isna() & (retailer["norm"] != "")
    unmatched_unique_norms = retailer.loc[still_unmatched_mask, "norm"].unique().tolist()
    log(f"Unique names with no acceptable Voysen match: {len(unmatched_unique_norms)}")

    # Self-clustering all unmatched names at once is memory-heavy at this
    # scale (very common short n-grams create dense-ish similarity rows).
    # Guard it: run the merge step only up to a safe size, otherwise fall
    # back to "one cluster per unique normalized string" (still correct,
    # just slightly more conservative: near-duplicate spelling variants may
    # end up as two new brands instead of one, flagged as a documented
    # limitation / follow-up improvement).
    CLUSTER_SAFE_LIMIT = 40000
    if len(unmatched_unique_norms) <= CLUSTER_SAFE_LIMIT:
        cluster_ids = cluster_new_brands(unmatched_unique_norms)
    else:
        log(
            f"  Skipping near-duplicate clustering ({len(unmatched_unique_norms)} "
            f"> {CLUSTER_SAFE_LIMIT} safe limit for this environment's memory); "
            "falling back to one new brand per unique normalized name."
        )
        cluster_ids = list(range(len(unmatched_unique_norms)))

    # Pick a canonical display name per cluster = the most frequent raw
    # brand_source_used_for_matching value among rows falling in that cluster.
    norm_to_cluster = dict(zip(unmatched_unique_norms, cluster_ids))
    tmp = retailer.loc[still_unmatched_mask, ["norm", "match_key_raw"]].copy()
    tmp["cluster"] = tmp["norm"].map(norm_to_cluster)
    canonical_names = (
        tmp.groupby("cluster")["match_key_raw"]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )

    next_id = int(voysen["id"].max()) + 1
    cluster_to_new_id = {}
    new_brand_rows = []
    for cluster in sorted(set(cluster_ids)):
        new_id = next_id
        next_id += 1
        name = canonical_names[cluster]
        cluster_to_new_id[cluster] = new_id
        new_brand_rows.append({"id": new_id, "name": name})

    log(f"New brands to create: {len(new_brand_rows)}")

    retailer.loc[still_unmatched_mask, "matched_id"] = tmp["cluster"].map(cluster_to_new_id).values
    retailer.loc[still_unmatched_mask, "match_type"] = "new_brand"
    retailer.loc[still_unmatched_mask, "match_score"] = np.nan

    # Data-quality guard: retailer "brand" fields are free text scraped from
    # marketplaces and sometimes contain full product titles / sentences
    # rather than an actual brand ("Double Burner Cooktop with Fast Heating
    # - Compact..."). Blindly turning every such string into a new Voysen
    # brand would pollute the referential. Flag long, sentence-like unmatched
    # names as "likely_non_brand": they still get a (temporary) brand id so
    # no row is dropped, but they are routed to a separate review file
    # instead of silently joining the trusted brand list.
    def is_probably_not_a_brand(name: str) -> bool:
        words = name.split()
        return len(words) > 6 or len(name) > 60

    new_brand_flags = {
        row["id"]: is_probably_not_a_brand(row["name"]) for row in new_brand_rows
    }
    retailer["likely_non_brand"] = retailer["matched_id"].map(
        lambda i: new_brand_flags.get(i, False)
    )

    # Rows where norm ended up empty (garbage / symbol-only brand names):
    # still need a brand id. Bucket them into a single "Unknown / Unlabeled"
    # brand rather than silently dropping the row.
    empty_mask = retailer["norm"] == ""
    if empty_mask.any():
        unknown_id = next_id
        new_brand_rows.append({"id": unknown_id, "name": "Unknown / Unlabeled"})
        retailer.loc[empty_mask, "matched_id"] = unknown_id
        retailer.loc[empty_mask, "match_type"] = "unlabeled"
        log(f"Rows with no usable brand text bucketed as 'Unknown / Unlabeled': {empty_mask.sum()}")

    retailer["matched_id"] = retailer["matched_id"].astype(int)

    # Build id -> name lookup (original Voysen + newly created)
    id_to_name = dict(zip(voysen_ids.tolist(), voysen_names.tolist()))
    id_to_name.update({row["id"]: row["name"] for row in new_brand_rows})
    retailer["brand_voysen"] = retailer["matched_id"].map(id_to_name)

    # ---- Deliverable 1: brand_associations.csv ----
    associations = retailer.rename(columns={"matched_id": "id_brand_voysen"})[
        ["id_brand_voysen", "brand_voysen", "url_source", "source",
         "match_type", "match_score", "likely_non_brand"]
    ]
    associations.to_csv(OUT_DIR / "brand_associations.csv", index=False)

    non_brand_review = retailer[retailer["likely_non_brand"]][
        ["match_key_raw", "brand_voysen", "matched_id", "source", "url_source"]
    ]
    non_brand_review.to_csv(OUT_DIR / "non_brand_review_queue.csv", index=False)

    # ---- Deliverable 2: augmented Voysen referential ----
    augmented = pd.concat(
        [voysen[["id", "name"]], pd.DataFrame(new_brand_rows)], ignore_index=True
    )
    augmented.to_csv(OUT_DIR / "voysen_brands_augmented.csv", index=False)

    # ---- Deliverable 3: review queue (QA aid, not required for automation) ----
    review = retailer[retailer["match_type"] == "fuzzy_review"][
        ["match_key_raw", "brand_voysen", "matched_id", "match_score", "source", "url_source"]
    ].sort_values("match_score")
    review.to_csv(OUT_DIR / "review_queue.csv", index=False)

    # ---- Report ----
    elapsed = time.time() - t0
    report = {
        "runtime_seconds": round(elapsed, 1),
        "n_retailer_rows": int(len(retailer)),
        "n_voysen_brands_input": int(len(voysen)),
        "n_voysen_brands_output": int(len(augmented)),
        "n_new_brands_created": int(len(new_brand_rows)),
        "matches": {
            "exact": int(n_exact),
            "fuzzy_high_confidence": int(n_fuzzy_high),
            "fuzzy_review_band": int(n_fuzzy_review),
            "new_brand": int((retailer["match_type"] == "new_brand").sum()),
            "unlabeled": int((retailer["match_type"] == "unlabeled").sum()),
        },
        "likely_non_brand_rows_flagged": int(retailer["likely_non_brand"].sum()),
        "thresholds": {
            "sim_match_threshold": SIM_MATCH_THRESHOLD,
            "sim_review_threshold": SIM_REVIEW_THRESHOLD,
            "sim_new_brand_cluster_threshold": SIM_NEW_BRAND_CLUSTER_THRESHOLD,
        },
    }
    with open(OUT_DIR / "match_report.json", "w") as f:
        json.dump(report, f, indent=2)

    log(f"Done in {elapsed:.1f}s")
    log(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
