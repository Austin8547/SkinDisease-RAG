import sys
import os
import time
import statistics

# Add src to Python path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    )
)

from retrieval import retrieve, retrieve_images
from rag import generate_answer


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
# Benchmark retrieval
# --------------------------------------------------

def benchmark_retrieval(query, top_k=5, max_images=3):

    # -------------------------------
    # Text retrieval
    # -------------------------------

    start = time.perf_counter()

    results = retrieve(
        query,
        top_k=top_k
    )

    text_latency = time.perf_counter() - start

    # -------------------------------
    # Image retrieval
    # -------------------------------

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
        "retrieval_latency": text_latency + image_latency,
        "num_chunks": len(results),
        "num_images": len(images)
    }


# --------------------------------------------------
# Benchmark complete RAG
# --------------------------------------------------

def benchmark_rag(query, max_images=3):

    start = time.perf_counter()

    result = generate_answer(
        query,
        max_images=max_images
    )

    total_latency = time.perf_counter() - start

    return {
        "total_latency": total_latency,
        "answer_length": len(result["answer"]),
        "num_images": len(result["images"])
    }


# --------------------------------------------------
# Main benchmark
# --------------------------------------------------

def main():

    print("=" * 70)
    print("RAG LATENCY EVALUATION")
    print("=" * 70)

    retrieval_times = []
    total_times = []

    for i, query in enumerate(QUERIES, 1):

        print(f"\nQuery {i}: {query}")
        print("-" * 70)

        # -------------------------------
        # Retrieval benchmark
        # -------------------------------

        retrieval = benchmark_retrieval(query)

        print(
            f"Text Retrieval : "
            f"{retrieval['text_latency'] * 1000:.2f} ms"
        )

        print(
            f"Image Retrieval: "
            f"{retrieval['image_latency'] * 1000:.2f} ms"
        )

        print(
            f"Total Retrieval: "
            f"{retrieval['retrieval_latency'] * 1000:.2f} ms"
        )

        print(
            f"Chunks Retrieved: "
            f"{retrieval['num_chunks']}"
        )

        print(
            f"Images Retrieved: "
            f"{retrieval['num_images']}"
        )

        retrieval_times.append(
            retrieval["retrieval_latency"]
        )

        # -------------------------------
        # Full RAG benchmark
        # -------------------------------

        rag = benchmark_rag(query)

        print(
            f"End-to-End RAG : "
            f"{rag['total_latency'] * 1000:.2f} ms"
        )

        total_times.append(
            rag["total_latency"]
        )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("LATENCY SUMMARY")
    print("=" * 70)

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

    print_stats(
        "Retrieval Latency",
        retrieval_times
    )

    print_stats(
        "End-to-End RAG Latency",
        total_times
    )


if __name__ == "__main__":
    main()