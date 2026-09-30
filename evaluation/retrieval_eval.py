import json
import os
import sys


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_PATH = os.path.join(
    PROJECT_ROOT,
    "src"
)

sys.path.insert(0, SRC_PATH)

from retrieval import retrieve


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "rag_evaluation_dataset-v2.json"
)

TOP_K = 5


# ============================================================
# LOAD DATASET
# ============================================================

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as f:

    dataset = json.load(f)


# ============================================================
# NORMALIZE DOCUMENT NAME
# ============================================================

def normalize_name(name):

    name = os.path.basename(name)

    name = name.lower().strip()

    if name.endswith(".pdf"):
        name = name[:-4]

    return name


# ============================================================
# METRIC STORAGE
# ============================================================

hits = []
precisions = []
recalls = []
reciprocal_ranks = []

failed_queries = []


# ============================================================
# EVALUATE EACH QUERY
# ============================================================

for item in dataset:

    question = item["question"]

    expected_source = normalize_name(
        item["source_document"]
    )

    # --------------------------------------------------------
    # Retrieve top-K chunks
    # --------------------------------------------------------

    results = retrieve(
        question,
        top_k=TOP_K
    )

    retrieved_sources = [
        normalize_name(
            result["source_pdf"]
        )
        for result in results
    ]

    # ========================================================
    # HIT@K
    # ========================================================

    if expected_source in retrieved_sources:

        hit = 1

    else:

        hit = 0

    hits.append(hit)

    # ========================================================
    # PRECISION@K
    # ========================================================

    relevant_count = sum(
        source == expected_source
        for source in retrieved_sources
    )

    precision = (
        relevant_count / TOP_K
    )

    precisions.append(precision)

    # ========================================================
    # RECALL@K
    # ========================================================
    #
    # There is only ONE relevant document in the dataset.
    #
    # Therefore:
    #
    # relevant documents retrieved
    # --------------------------------
    # total relevant documents
    #
    # = 1 / 1 = 1
    #
    # if the expected document is retrieved.
    #
    # Otherwise = 0.
    #
    # Therefore Recall@5 == Hit@5 for this dataset.
    # ========================================================

    relevant_documents_retrieved = (
        1 if expected_source in retrieved_sources else 0
    )

    total_relevant_documents = 1

    recall = (
        relevant_documents_retrieved
        / total_relevant_documents
    )

    recalls.append(recall)

    # ========================================================
    # MRR@K
    # ========================================================

    reciprocal_rank = 0

    for rank, source in enumerate(
        retrieved_sources,
        start=1
    ):

        if source == expected_source:

            reciprocal_rank = 1 / rank

            break

    reciprocal_ranks.append(
        reciprocal_rank
    )

    # ========================================================
    # STORE FAILED QUERIES
    # ========================================================

    if hit == 0:

        failed_queries.append({

            "id": item["id"],

            "question": question,

            "expected": expected_source,

            "retrieved": retrieved_sources
        })


# ============================================================
# CALCULATE FINAL METRICS
# ============================================================

num_queries = len(dataset)

hit_at_k = (
    sum(hits) / num_queries
)

precision_at_k = (
    sum(precisions) / num_queries
)

recall_at_k = (
    sum(recalls) / num_queries
)

mrr_at_k = (
    sum(reciprocal_ranks) / num_queries
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("RETRIEVAL PERFORMANCE EVALUATION")
print("=" * 60)

print(
    f"\nNumber of queries: {num_queries}"
)

print(
    f"Top-K: {TOP_K}"
)

print("\nPerformance Metrics:")

print(
    f"Hit@{TOP_K}:       "
    f"{hit_at_k:.4f} "
    f"({hit_at_k * 100:.2f}%)"
)

print(
    f"Precision@{TOP_K}: "
    f"{precision_at_k:.4f} "
    f"({precision_at_k * 100:.2f}%)"
)

print(
    f"Recall@{TOP_K}:    "
    f"{recall_at_k:.4f} "
    f"({recall_at_k * 100:.2f}%)"
)

print(
    f"MRR@{TOP_K}:       "
    f"{mrr_at_k:.4f}"
)

