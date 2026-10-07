import json
from pathlib import Path

# pyrefly: ignore [missing-import]
import chromadb

from config import EMBEDDING_PATH, CHROMA_PATH  
# --------------------------------------------------
# 1. File paths
# --------------------------------------------------
input_file = Path(EMBEDDING_PATH)
chroma_path = Path(CHROMA_PATH)


# --------------------------------------------------
# 2. Load embedded chunks
# --------------------------------------------------

with open(input_file, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} embedded chunks")


# --------------------------------------------------
# 3. Create persistent Chroma client
# --------------------------------------------------

client = chromadb.PersistentClient(
    path=str(chroma_path)
)


# --------------------------------------------------
# 4. Create / get collection
# --------------------------------------------------

collection = client.get_or_create_collection(
    name="skin_disease"
)


# --------------------------------------------------
# 5. Prepare data
# --------------------------------------------------

ids = []
documents = []
embeddings = []
metadatas = []

for chunk in chunks:

    ids.append(str(chunk["chunk_id"]))

    documents.append(chunk["text"])

    embeddings.append(chunk["embedding"])

    metadatas.append({
        "disease": chunk["disease"],
        "source_pdf": chunk["source_pdf"]
    })


# --------------------------------------------------
# 6. Add to ChromaDB
# --------------------------------------------------

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)


# --------------------------------------------------
# 7. Check database
# --------------------------------------------------

print("Vector database created successfully!")
print("Collection:", collection.name)
print("Number of vectors:", collection.count())
print("Database location:", chroma_path)