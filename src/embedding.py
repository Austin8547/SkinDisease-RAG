import json
from pathlib import Path
from sentence_transformers import SentenceTransformer


from config import CHUNKS_PATH, EMBEDDING_PATH, EMBEDDING_MODEL



input_file = Path(CHUNKS_PATH)
output_file = Path(EMBEDDING_PATH)


# 1. Load chunks
# --------------------------------------------------

with open(input_file, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")


# --------------------------------------------------
# 3. Load embedding model
# --------------------------------------------------

model = SentenceTransformer(
    EMBEDDING_MODEL,
    device="cuda"
)

print("Embedding model loaded")


# --------------------------------------------------
# 4. Extract text from chunks
# --------------------------------------------------

texts = [chunk["text"] for chunk in chunks]


# --------------------------------------------------
# 5. Generate embeddings
# --------------------------------------------------

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Embeddings generated")
print("Embedding shape:", embeddings.shape)


# --------------------------------------------------
# 6. Create output data
# --------------------------------------------------

embedded_chunks = []

for chunk, embedding in zip(chunks, embeddings):

    embedded_chunks.append({
        "chunk_id": chunk["chunk_id"],
        "disease": chunk["disease"],
        "source_pdf": chunk["source_pdf"],
        "text": chunk["text"],
        "embedding": embedding.tolist()
    })


# --------------------------------------------------
# 7. Save embeddings
# --------------------------------------------------

output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(
        embedded_chunks,
        f,
        indent=4,
        ensure_ascii=False
    )


print(f"Saved embeddings to: {output_file}")