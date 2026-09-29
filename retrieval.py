import json
import os

import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Embedding model
# --------------------------------------------------

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    device="cuda"
)


# --------------------------------------------------
# ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="/home/austin/agentic/skin_rag/data/chroma_db"
)

collection = client.get_collection(
    name="skin_disease"
)


# --------------------------------------------------
# Image metadata
# --------------------------------------------------

IMAGE_METADATA_PATH = (
    "/home/austin/agentic/skin_rag/data/meta_data.json"
)

with open(IMAGE_METADATA_PATH, "r") as f:
    image_metadata = json.load(f)


# --------------------------------------------------
# Text retrieval
# --------------------------------------------------

def retrieve(query, top_k=5):

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

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


# --------------------------------------------------
# Image retrieval
# --------------------------------------------------

def retrieve_images(source_pdf, max_images=5):

    matched_images = []

    # Remove .pdf from source name
    source_name = os.path.splitext(
        os.path.basename(source_pdf)
    )[0]

    for image in image_metadata:

        image_disease = image["disease"]

        # Match PDF/source with image metadata
        if (
            source_name.lower() == image_disease.lower()
            or
            source_name.lower() in image_disease.lower()
            or
            image_disease.lower() in source_name.lower()
        ):

            matched_images.append(image)

            if len(matched_images) >= max_images:
                break

    return matched_images


