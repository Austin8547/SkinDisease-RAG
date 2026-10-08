# Skin Disease Multimodal RAG

A multimodal Retrieval-Augmented Generation (RAG) assistant that answers clinical dermatology questions using 100 DermNet guides and retrieves relevant clinical reference images alongside text answers.

---

## What We Did

1. **PDF Ingestion & Extraction**: Extracted raw clinical text and images from 100 DermNet PDF documents (`dermnetfiles/`).
2. **Text Cleaning & Image Mapping**: Cleaned and formatted medical texts, organized extracted images by disease, and built image metadata mappings (`data/meta_data.json`).
3. **Chunking & Embeddings**: Chunked clinical text (500 chars, 50 overlap) and computed dense vector embeddings using `BAAI/bge-base-en-v1.5`.
4. **Vector Database**: Stored chunk embeddings in ChromaDB for fast semantic similarity search.
5. **Multimodal Retrieval**: Built retrieval logic to return both top relevant text passages and matched clinical images.
6. **Agentic RAG Pipeline**: Built a LangGraph agent powered by Groq (`qwen/qwen3.8-27b`) with memory checkpointer, conversation summarization middleware, and retrieval tools.
7. **Web Interface**: Developed an interactive Streamlit UI supporting multi-session chat, conversation history, and side-by-side image display.
8. **Evaluation & Benchmarking**: Evaluated retrieval performance (Hit@K, MRR, Precision, Recall) and measured system latencies.

---

## What's Inside

```text
skin_rag/
├── data/                  # Cleaned texts, chunks, embeddings, metadata, and evaluation datasets
├── dermnetfiles/          # 100 original DermNet clinical guide PDFs
├── images/                # Extracted clinical photos organized by disease
├── chroma_db/             # Persistent ChromaDB vector database
├── src/
│   ├── pdf_processing/    # PDF text/image extraction, cleaning, and metadata generation
│   ├── rag/               # LangGraph agent, Groq LLM setup, memory, and tool definitions
│   ├── app.py             # Streamlit web application
│   ├── chroma_db.py       # ChromaDB index creation script
│   ├── chunk_text.py      # Text chunking script
│   ├── config.py          # Central configuration (paths, chunk sizes, models)
│   ├── embedding.py       # Embedding generation script
│   └── retrieval.py       # Text and image retrieval functions
├── evaluation/            # Scripts to evaluate retrieval metrics and latency
├── testing/               # Integration and retrieval test scripts
├── requirement.txt        # Python package dependencies
└── README.md              # Project documentation
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirement.txt
```

### 2. Set API Key
```bash
export GROQ_API_KEY="your-groq-api-key"
```

### 3. Run the App
```bash
streamlit run src/app.py
```
