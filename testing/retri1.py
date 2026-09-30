import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# Import retrieval functions from your main script
# --------------------------------------------------
# Assumes your original code is saved in `rag_retriever.py`
try:
    from retrieval import retrieve, retrieve_images, collection, image_metadata
except ImportError:
    print("❌ Error: Make sure your original code is saved as 'rag_retriever.py' in the same directory.")
    exit(1)


def check_database_status():
    """Verify ChromaDB collection status."""
    print("=" * 60)
    print("1. CHECKING CHROMADB STATUS")
    print("=" * 60)
    
    count = collection.count()
    print(f"✓ Connected to collection: '{collection.name}'")
    print(f"✓ Total documents stored in DB: {count}")
    
    if count == 0:
        print("⚠️ Warning: ChromaDB collection is empty! Text retrieval will return no results.")
    print()


def check_image_metadata_status():
    """Verify JSON metadata loading."""
    print("=" * 60)
    print("2. CHECKING IMAGE METADATA STATUS")
    print("=" * 60)
    
    total_images = len(image_metadata)
    print(f"✓ Image metadata loaded successfully.")
    print(f"✓ Total image records in JSON: {total_images}")
    
    if total_images > 0:
        print("\nSample image record structure:")
        print(json.dumps(image_metadata[0], indent=2))
    else:
        print("⚠️ Warning: Image metadata JSON file is empty!")
    print()


def test_text_retrieval(sample_query):
    """Test text search query."""
    print("=" * 60)
    print(f"3. TESTING TEXT RETRIEVAL")
    print(f"   Query: '{sample_query}'")
    print("=" * 60)
    
    results = retrieve(sample_query, top_k=3)
    
    if not results:
        print("No chunks retrieved.")
        return None
    
    for idx, item in enumerate(results, 1):
        print(f"\n--- Result #{idx} ---")
        print(f"Disease    : {item['disease']}")
        print(f"Source PDF : {item['source_pdf']}")
        print(f"Distance   : {item['distance']:.4f}")
        # Print snippet of retrieved text
        snippet = item['text'][:150].replace('\n', ' ')
        print(f"Text Chunk : {snippet}...")
        
    return results[0]["source_pdf"] if results else None


def test_image_retrieval(source_pdf):
    """Test image retrieval using a source PDF name."""
    print("\n" + "=" * 60)
    print(f"4. TESTING IMAGE RETRIEVAL")
    print(f"   Source PDF target: '{source_pdf}'")
    print("=" * 60)
    
    matched_images = retrieve_images(source_pdf, max_images=3)
    
    if not matched_images:
        print(f"❌ No matching images found in JSON for source: {source_pdf}")
    else:
        print(f"✓ Found {len(matched_images)} matching image record(s):\n")
        for idx, img in enumerate(matched_images, 1):
            print(f"  [{idx}] Disease Field: {img.get('disease')}")
            print(f"      Image Path   : {img.get('image_path', img.get('file_name', 'N/A'))}")
    print()


if __name__ == "__main__":
    # Step 1: Check database loading
    check_database_status()
    
    # Step 2: Check metadata loading
    check_image_metadata_status()
    
    # Step 3: Run end-to-end retrieval check
    test_query = "red rashes and itching on skin"
    
    top_source_pdf = test_text_retrieval(sample_query=test_query)
    
    # Step 4: Run image check based on top retrieved PDF source
    if top_source_pdf:
        test_image_retrieval(top_source_pdf)
    else:
        # Fallback manual check if DB is empty
        test_image_retrieval("eczema.pdf")