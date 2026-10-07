"""
Simple end-to-end RAG test.

Run from project root:
    python3 -m src.testing.rag_test
"""

from src.rag.chain import generate_answer

# --------------------------------------------------
# Test queries
# --------------------------------------------------

QUERIES = [
    "What causes acne?",
    "What are the symptoms of psoriasis?",
    "How is vitiligo treated?",
]


# --------------------------------------------------
# Run tests
# --------------------------------------------------

def test_rag(query: str, session_id: str = "test"):

    print("\n" + "=" * 60)
    print(f"Query: {query}")
    print("=" * 60)

    result = generate_answer(
        query=query,
        session_id=session_id,
        max_images=2
    )

    answer = result.get("answer", "")
    images = result.get("images", [])

    print(f"\nAnswer ({len(answer)} chars):")
    print(answer[:500] + ("..." if len(answer) > 500 else ""))

    print(f"\nImages retrieved: {len(images)}")
    for img in images:
        print(f"  - {img.get('disease')} | {img.get('image_path', 'N/A')}")


def main():
    print("=" * 60)
    print("RAG PIPELINE TEST")
    print("=" * 60)

    for i, query in enumerate(QUERIES, 1):
        test_rag(query, session_id=f"test_session_{i}")

    print("\n" + "=" * 60)
    print("All tests complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
