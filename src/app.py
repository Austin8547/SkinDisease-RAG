import streamlit as st
import os

# Page configuration
st.set_page_config(
    page_title="Skin Disease RAG Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 1.25rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .img-card {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px;
        background-color: #FFFFFF;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Cache model/imports so Streamlit doesn't reload heavy dependencies on every click
@st.cache_resource
def load_rag_pipeline():
    from rag import generate_answer
    return generate_answer

generate_answer = load_rag_pipeline()

# --------------------------------------------------
# Sidebar Configuration
# --------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/387/387561.png", width=70)
    st.title("RAG Configuration")
    
    max_images = st.slider("Max Images to Retrieve", min_value=1, max_value=6, value=3)
    
    st.markdown("---")
    st.markdown("### System Details")
    st.markdown("- **Embedding:** `BAAI/bge-base-en-v1.5`")
    st.markdown("- **Vector Store:** ChromaDB")
    st.markdown("- **LLM:** Groq (`qwen3.8-27b`)")
    
    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** This tool is for educational/informational purposes only. Consult a healthcare professional for clinical diagnostics.")

# --------------------------------------------------
# Main UI Layout
# --------------------------------------------------
st.markdown('<div class="main-header">🩺 Skin Disease RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Diagnostic Information & Visual Reference System</div>', unsafe_allow_html=True)

# Query Input Form
with st.form(key="search_form"):
    user_query = st.text_input(
        "Enter your question or symptoms:",
        placeholder="e.g., What causes acanthosis nigricans? or What are the signs of melanoma?",
    )
    submit_button = st.form_submit_button(label="🔍 Search Knowledge Base", use_container_width=True)

# Sample Query Chips
st.markdown("**Example Questions:**")
col_e1, col_e2, col_e3 = st.columns(3)
if col_e1.button("What causes acanthosis nigricans?"):
    user_query = "What causes acanthosis nigricans?"
    submit_button = True
if col_e2.button("How to treat atopic dermatitis?"):
    user_query = "How to treat atopic dermatitis?"
    submit_button = True
if col_e3.button("Symptoms of psoriasis"):
    user_query = "Symptoms of psoriasis"
    submit_button = True

# --------------------------------------------------
# Results Execution
# --------------------------------------------------
if submit_button and user_query:
    with st.spinner("Searching medical records & processing answer..."):
        try:
            # Execute RAG Pipeline
            result = generate_answer(user_query, max_images=max_images)
            
            st.markdown("### 📋 Medical Insights")
            
            # Display LLM Text Response
            st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

            # Display Retrieved Images
            st.markdown("### 📸 Visual References")
            images = result.get("images", [])

            if images:
                # Create grid columns dynamically for images
                cols = st.columns(min(len(images), 3))
                
                for idx, img in enumerate(images):
                    col = cols[idx % 3]
                    
                    # Try retrieving image path key flexible to common schema names
                    img_path = img.get("image_path") or img.get("file_path") or img.get("file_name") or img.get("url")
                    disease_label = img.get("disease", "Condition Reference")

                    with col:
                        st.markdown('<div class="img-card">', unsafe_allow_html=True)
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, caption=f"{disease_label}", use_container_width=True)
                        elif img_path and (img_path.startswith("http://") or img_path.startswith("https://")):
                            st.image(img_path, caption=f"{disease_label}", use_container_width=True)
                        else:
                            # Display metadata placeholder if local image file isn't found at exact path
                            st.warning(f"🖼️ **Image Metadata Found**\n\n**Disease:** {disease_label}\n\n*File:* `{img_path}`")
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No matching visual reference images were found for this query.")

        except Exception as e:
            st.error(f"An error occurred while generating the answer: {str(e)}")