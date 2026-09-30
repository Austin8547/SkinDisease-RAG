import sys
import os
import time
import statistics

# --------------------------------------------------
# Add src to Python path
# --------------------------------------------------

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    )
)

from retrieval import retrieve, retrieve_images
from rag import llm


# --------------------------------------------------
# Test queries
# --------------------------------------------------

QUERIES = [
    "What causes acanthosis nigricans?",
    "What are the symptoms of psoriasis?",
    "How is atopic dermatitis treated?",
    "What are the symptoms of melanoma?",
    "What causes vitiligo?"
]


# --------------------------------------------------
# Retrieval benchmark
# --------------------------------------------------

def benchmark_retrieval(query, top_k=5, max_images=3):

    # ----------------------------------------------
    # Text retrieval
    # ----------------------------------------------

    start = time.perf_counter()

    results = retrieve(
        query,
        top_k=top_k
    )

    text_latency = time.perf_counter() - start

    # ----------------------------------------------
    # Image retrieval
    # ----------------------------------------------

    start = time.perf_counter()

    images = []
    seen_sources = set()

    for result in results:

        source_pdf = result["source_pdf"]

        if source_pdf not in seen_sources:

            seen_sources.add(source_pdf)

            matched = retrieve_images(
                source_pdf,
                max_images=max_images
            )

            images.extend(matched)

        if len(images) >= max_images:

            images = images[:max_images]

            break

    image_latency = time.perf_counter() - start

    return {
        "text_latency": text_latency,
        "image_latency": image_latency,
        "retrieval_latency": (
            text_latency + image_latency
        ),
        "num_chunks": len(results),
        "num_images": len(images)
    }


# --------------------------------------------------
# LLM benchmark
# --------------------------------------------------

def benchmark_llm(query, top_k=5):

    # ----------------------------------------------
    # Retrieve context
    # ----------------------------------------------

    results = retrieve(
        query,
        top_k=top_k
    )

    if not results:
        return {
            "llm_latency": 0,
            "total_latency": 0,
            "answer_length": 0
        }

    # ----------------------------------------------
    # Build context
    # ----------------------------------------------

    context = ""

    for result in results:

        context += f"""
Disease: {result["disease"]}
Source: {result["source_pdf"]}

{result["text"]}

"""

    # ----------------------------------------------
    # Build prompt
    # ----------------------------------------------

    prompt = f"""
You are a medical information assistant.

Answer the user's question using the provided context.

If the answer is not available in the context, say that
the information is not available in the provided knowledge base.

Context:
{context}

Question:
{query}

Answer:
"""

    # ----------------------------------------------
    # LLM latency
    # ----------------------------------------------

    start = time.perf_counter()

    response = llm.invoke(prompt)

    llm_latency = time.perf_counter() - start

    return {
        "llm_latency": llm_latency,
        "total_latency": llm_latency,
        "answer_length": len(response.content)
    }


# --------------------------------------------------
# Warm-up
# --------------------------------------------------

def warmup():

    print("\n" + "=" * 70)
    print("WARM-UP")
    print("=" * 70)

    warmup_queries = [
        "What is psoriasis?",
        "What causes eczema?",
        "What are the symptoms of melanoma?"
    ]

    for query in warmup_queries:

        print(f"Warming up: {query}")

        retrieve(
            query,
            top_k=5
        )

    print("Warm-up complete.")


# --------------------------------------------------
# Statistics
# --------------------------------------------------

def percentile(values, p):

    values = sorted(values)

    if not values:
        return 0

    index = (len(values) - 1) * p

    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    weight = index - lower

    return (
        values[lower]
        + weight * (values[upper] - values[lower])
    )


def print_stats(name, values):

    values_ms = [
        value * 1000
        for value in values
    ]

    print(f"\n{name}")

    print(
        f"Mean   : "
        f"{statistics.mean(values_ms):.2f} ms"
    )

    print(
        f"Median : "
        f"{statistics.median(values_ms):.2f} ms"
    )

    print(
        f"P95    : "
        f"{percentile(values_ms, 0.95):.2f} ms"
    )

    print(
        f"Min    : "
        f"{min(values_ms):.2f} ms"
    )

    print(
        f"Max    : "
        f"{max(values_ms):.2f} ms"
    )

    if len(values_ms) >= 2:

        print(
            f"Std Dev: "
            f"{statistics.stdev(values_ms):.2f} ms"
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("RAG LATENCY EVALUATION")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Measure cold-start retrieval
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("COLD-START RETRIEVAL")
    print("=" * 70)

    cold_query = QUERIES[0]

    start = time.perf_counter()

    cold_result = retrieve(
        cold_query,
        top_k=5
    )

    cold_latency = time.perf_counter() - start

    print(f"\nQuery: {cold_query}")

    print(
        f"Cold-start retrieval: "
        f"{cold_latency * 1000:.2f} ms"
    )

    print(
        f"Chunks retrieved: "
        f"{len(cold_result)}"
    )

    # --------------------------------------------------
    # 2. Warm-up
    # --------------------------------------------------

    warmup()

    # --------------------------------------------------
    # 3. Warm retrieval benchmark
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("WARM RETRIEVAL BENCHMARK")
    print("=" * 70)

    retrieval_times = []
    text_times = []
    image_times = []

    for i, query in enumerate(QUERIES, 1):

        print("\n" + "-" * 70)
        print(f"Query {i}: {query}")
        print("-" * 70)

        result = benchmark_retrieval(
            query,
            top_k=5,
            max_images=3
        )

        text_ms = (
            result["text_latency"] * 1000
        )

        image_ms = (
            result["image_latency"] * 1000
        )

        total_ms = (
            result["retrieval_latency"] * 1000
        )

        print(
            f"Text Retrieval : "
            f"{text_ms:.2f} ms"
        )

        print(
            f"Image Retrieval: "
            f"{image_ms:.2f} ms"
        )

        print(
            f"Total Retrieval: "
            f"{total_ms:.2f} ms"
        )

        print(
            f"Chunks Retrieved: "
            f"{result['num_chunks']}"
        )

        print(
            f"Images Retrieved: "
            f"{result['num_images']}"
        )

        text_times.append(
            result["text_latency"]
        )

        image_times.append(
            result["image_latency"]
        )

        retrieval_times.append(
            result["retrieval_latency"]
        )

    # --------------------------------------------------
    # 4. LLM benchmark
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("LLM LATENCY BENCHMARK")
    print("=" * 70)

    llm_times = []

    for i, query in enumerate(QUERIES, 1):

        print("\n" + "-" * 70)
        print(f"Query {i}: {query}")
        print("-" * 70)

        result = benchmark_llm(
            query,
            top_k=5
        )

        llm_ms = (
            result["llm_latency"] * 1000
        )

        print(
            f"LLM Generation: "
            f"{llm_ms:.2f} ms"
        )

        print(
            f"Answer length: "
            f"{result['answer_length']} characters"
        )

        llm_times.append(
            result["llm_latency"]
        )

    # --------------------------------------------------
    # 5. Summary
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("LATENCY SUMMARY")
    print("=" * 70)

    print(
        f"\nCold-start retrieval: "
        f"{cold_latency * 1000:.2f} ms"
    )

    print_stats(
        "Warm Text Retrieval",
        text_times
    )

    print_stats(
        "Warm Image Retrieval",
        image_times
    )

    print_stats(
        "Warm Total Retrieval",
        retrieval_times
    )

    print_stats(
        "LLM Latency",
        llm_times
    )


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()