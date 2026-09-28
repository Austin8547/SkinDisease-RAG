import chromadb
from sentence_transformers import SentenceTransformer


# -----------------------------
# 1. Load embedding model
# -----------------------------

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    device="cuda"
)


# -----------------------------
# 2. Connect to ChromaDB
# -----------------------------

client = chromadb.PersistentClient(
    path="/home/austin/agentic/skin_rag/dataset/chroma_db"
)

collection = client.get_collection(
    name="skin_disease"
)


# -----------------------------
# 3. Retrieval function
# -----------------------------

def retrieve(query, top_k=5):

    # Convert query into embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved_chunks = []

    for i in range(len(results["documents"][0])):

        retrieved_chunks.append({
            "text": results["documents"][0][i],
            "disease": results["metadatas"][0][i]["disease"],
            "source_pdf": results["metadatas"][0][i]["source_pdf"],
            "distance": results["distances"][0][i]
        })

    return retrieved_chunks


# -----------------------------
# 4. Test retrieval
# -----------------------------

if __name__ == "__main__":

    query = "What causes acanthosis nigricans?"

    results = retrieve(
        query,
        top_k=5
    )

    print("\nQuery:")
    print(query)

    print("\nRetrieved results:")

    for i, result in enumerate(results):

        print("\n" + "=" * 70)
        print(f"Result {i + 1}")
        print("=" * 70)

        print("Disease:", result["disease"])
        print("Source:", result["source_pdf"])
        print("Distance:", result["distance"])

        print("\nText:")
        print(result["text"][:500])