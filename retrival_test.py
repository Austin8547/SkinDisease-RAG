import chromadb

from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Load embedding model
# --------------------------------------------------

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    device="cuda"
)


# --------------------------------------------------
# 2. Connect to existing ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="/home/austin/agentic/skin_rag/dataset/chroma_db"
)

collection = client.get_collection(
    name="skin_disease"
)


# --------------------------------------------------
# 3. User query
# --------------------------------------------------

query = "What causes acne?"


# --------------------------------------------------
# 4. Convert query into embedding
# --------------------------------------------------

query_embedding = model.encode(
    query,
    normalize_embeddings=True
).tolist()


# --------------------------------------------------
# 5. Search ChromaDB
# --------------------------------------------------

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)


# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

for i in range(len(results["documents"][0])):

    print("\n" + "=" * 60)

    print("Result:", i + 1)

    print("Disease:", results["metadatas"][0][i]["disease"])

    print("Source:", results["metadatas"][0][i]["source_pdf"])

    print("Distance:", results["distances"][0][i])

    print("\nText:")
    print(results["documents"][0][i])