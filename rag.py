from retrival import retrieve
from langchain_groq import ChatGroq


llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0
)


def generate_answer(query):

    results = retrieve(query, top_k=5)

    context = ""

    for result in results:
        context += f"""
Disease: {result["disease"]}
Source: {result["source_pdf"]}

{result["text"]}

"""


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

    response = llm.invoke(prompt)

    return response.content


if __name__ == "__main__":

    query = "What causes acanthosis nigricans?"

    answer = generate_answer(query)

    print("\nAnswer:")
    print(answer)