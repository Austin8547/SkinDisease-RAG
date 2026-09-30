from retrieval import retrieve, retrieve_images
from langchain_groq import ChatGroq

# Initialize LLM
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0
)


def generate_answer(query, max_images=3):
    # 1. Retrieve text chunks from ChromaDB
    results = retrieve(query, top_k=5)

    if not results:
        return {
            "answer": "No relevant documents were found in the knowledge base.",
            "images": []
        }

    # 2. Build text context for LLM
    context = ""
    for result in results:
        context += f"""
Disease: {result["disease"]}
Source: {result["source_pdf"]}

{result["text"]}

"""

    # 3. Retrieve relevant images using source_pdf from top text results
    images = []
    seen_sources = set()

    for result in results:
        source_pdf = result["source_pdf"]

        # Avoid searching the exact same PDF multiple times
        if source_pdf not in seen_sources:
            seen_sources.add(source_pdf)
            matched = retrieve_images(source_pdf, max_images=max_images)
            images.extend(matched)

        # Stop collecting images once target threshold is reached
        if len(images) >= max_images:
            images = images[:max_images]
            break

    # 4. Construct Prompt
    prompt = f"""
You are a medical information assistant.

Answer the user's question using the provided context.

If the answer is not available in the context, say that
the information is not available in the provided knowledge base.

Context:
{context}

Question:
{query}

Answer:
"""

    # 5. Generate Answer via LLM
    response = llm.invoke(prompt)

    # Return structured dict with text answer and matching images
    return {
        "answer": response.content,
        "images": images
    }


if __name__ == "__main__":
    query = "What causes acanthosis nigricans?"

    result = generate_answer(query)

    print("\n" + "=" * 50)
    print("ANSWER:")
    print("=" * 50)
    print(result["answer"])

    print("\n" + "=" * 50)
    print("RETRIEVED IMAGES:")
    print("=" * 50)
    if result["images"]:
        for idx, img in enumerate(result["images"], 1):
            print(f"[{idx}] Disease: {img.get('disease')}")
            print(f"    Path/URL: {img.get('image_path', img.get('file_name', 'N/A'))}")
    else:
        print("No matching images found in metadata.")