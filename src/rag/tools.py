import json

from retrieval import retrieve, retrieve_images


def search_skin_knowledge_base(query: str) -> str:
    """
    Search the skin disease knowledge base.
    """

    results = retrieve(query, top_k=5)

    if not results:
        return json.dumps({
            "context": "",
            "images": []
        })

    context = ""

    for result in results:
        context += f"""
Disease: {result["disease"]}
Source: {result["source_pdf"]}

{result["text"]}

"""

    images = []
    seen_sources = set()

    for result in results:

        source_pdf = result["source_pdf"]

        if source_pdf not in seen_sources:

            seen_sources.add(source_pdf)

            matched = retrieve_images(
                source_pdf,
                max_images=3
            )

            images.extend(matched)

        if len(images) >= 3:
            images = images[:3]
            break

    return json.dumps({
        "context": context,
        "images": images
    })