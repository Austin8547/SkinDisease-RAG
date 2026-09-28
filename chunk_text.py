import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


input_file = Path("/home/austin/agentic/skin_rag/dataset/clean_text.json")
output_file = Path("/home/austin/agentic/skin_rag/dataset/chunks.json")


# 1. Load cleaned JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)


# 2. Convert JSON records into LangChain Documents
documents = []

for item in data:
    documents.append(
        Document(
            page_content=item["text"],
            metadata={
                "disease": item["disease"],
                "source_pdf": item["source_pdf"]
            }
        )
    )


# 3. Create text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


# 4. Split documents
chunks = splitter.split_documents(documents)


# 5. Convert chunks back to JSON
chunk_data = []

for i, chunk in enumerate(chunks):

    chunk_data.append({
        "chunk_id": i + 1,
        "disease": chunk.metadata["disease"],
        "source_pdf": chunk.metadata["source_pdf"],
        "text": chunk.page_content
    })


# 6. Save chunks
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(chunk_data, f, indent=4, ensure_ascii=False)


print(f"Created {len(chunk_data)} chunks")
print(f"Saved to: {output_file}")