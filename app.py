import os
import streamlit as st

from rag import generate_answer
from retrieval import retrieve


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Skin Disease RAG",
    page_icon="🩺",
    layout="wide"
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

IMAGE_DIR = "dataset/images"


# --------------------------------------------------
# Helper function to get images
# --------------------------------------------------

def get_disease_images(disease, max_images=5):
    """
    Get images from the folder corresponding to the disease.
    """

    disease_folder = disease.lower().replace(" ", "_")

    folder_path = os.path.join(
        IMAGE_DIR,
        disease_folder
    )

    if not os.path.exists(folder_path):
        return []

    image_files = []

    for filename in os.listdir(folder_path):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            image_files.append(
                os.path.join(folder_path, filename)
            )

    return image_files[:max_images]


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🩺 Skin Disease RAG")

st.write(
    "Ask questions about the skin-disease knowledge base."
)


query = st.text_input(
    "Enter your question",
    placeholder="What causes acanthosis nigricans?"
)


# --------------------------------------------------
# Ask button
# --------------------------------------------------

if st.button("Ask"):

    if not query.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner(
            "Searching knowledge base and generating answer..."
        ):

            # Retrieve relevant text chunks
            results = retrieve(
                query,
                top_k=5
            )

            # Generate RAG answer
            answer = generate_answer(query)


        # --------------------------------------------------
        # Answer
        # --------------------------------------------------

        st.subheader("Answer")

        st.write(answer)


        # --------------------------------------------------
        # Relevant Images
        # --------------------------------------------------

        st.subheader("🖼️ Relevant Images")

        displayed_diseases = set()

        for result in results:

            disease = result["disease"]

            # Avoid displaying the same disease multiple times
            if disease in displayed_diseases:
                continue

            displayed_diseases.add(disease)

            images = get_disease_images(
                disease,
                max_images=5
            )

            if images:

                st.write(f"**{disease}**")

                # Create columns for images
                columns = st.columns(
                    min(len(images), 5)
                )

                for i, image_path in enumerate(images):

                    with columns[i]:
                        st.image(
                            image_path,
                            use_container_width=True
                        )

            else:

                st.info(
                    f"No images found for {disease}."
                )


        # --------------------------------------------------
        # Retrieved Sources
        # --------------------------------------------------

        st.subheader("📚 Retrieved Sources")

        for i, result in enumerate(results):

            with st.expander(
                f"Result {i + 1} — {result['disease']}"
            ):

                st.write(
                    f"**Source:** {result['source_pdf']}"
                )

                st.write(
                    f"**Distance:** {result['distance']}"
                )

                st.write(result["text"])