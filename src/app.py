import streamlit as st
import os
import uuid


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Skin Disease RAG Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Custom Styling
# ============================================================

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
    margin-bottom: 1rem;
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


# ============================================================
# Load RAG Pipeline
# ============================================================

@st.cache_resource
def load_rag_pipeline():

    from rag import (
        generate_answer,
        clear_conversation
    )

    return generate_answer, clear_conversation


generate_answer, clear_conversation = load_rag_pipeline()


# ============================================================
# Initialize Session State
# ============================================================

# ------------------------------------------------------------
# Store all chat sessions
# ------------------------------------------------------------

if "sessions" not in st.session_state:

    st.session_state.sessions = {}


# ------------------------------------------------------------
# Create first session
# ------------------------------------------------------------

if "session_id" not in st.session_state:

    session_id = str(uuid.uuid4())

    st.session_state.session_id = session_id

    st.session_state.sessions[session_id] = {
        "title": "New Chat",
        "messages": []
    }


# ------------------------------------------------------------
# Get messages of current session
# ------------------------------------------------------------

current_session = st.session_state.sessions[
    st.session_state.session_id
]

st.session_state.messages = current_session["messages"]


# ============================================================
# New Chat Function
# ============================================================

def start_new_chat():

    # Create a new session ID
    new_session_id = str(uuid.uuid4())

    # Create new session
    st.session_state.sessions[new_session_id] = {
        "title": "New Chat",
        "messages": []
    }

    # Make it the active session
    st.session_state.session_id = new_session_id

    # Point messages to new session
    st.session_state.messages = (
        st.session_state.sessions[new_session_id]["messages"]
    )


# ============================================================
# Switch Chat Function
# ============================================================

def switch_chat(session_id):

    # Change active session
    st.session_state.session_id = session_id

    # Load messages belonging to that session
    st.session_state.messages = (
        st.session_state.sessions[session_id]["messages"]
    )


# ============================================================
# Delete Chat Function
# ============================================================

def delete_chat(session_id):

    # Don't delete if it is the only session
    if len(st.session_state.sessions) <= 1:
        return

    # Try to clear LangGraph memory
    try:

        clear_conversation(session_id)

    except Exception:

        pass

    # Remove session from Streamlit history
    if session_id in st.session_state.sessions:

        del st.session_state.sessions[session_id]

    # If deleted session was active,
    # switch to another available session
    if session_id == st.session_state.session_id:

        new_session_id = next(
            iter(st.session_state.sessions)
        )

        st.session_state.session_id = new_session_id

        st.session_state.messages = (
            st.session_state.sessions[new_session_id]["messages"]
        )


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/387/387561.png",
        width=70
    )

    st.title("RAG Configuration")


    # ========================================================
    # New Chat
    # ========================================================

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        start_new_chat()

        st.rerun()


    st.markdown("---")


    # ========================================================
    # Chat History
    # ========================================================

    st.markdown("### 💬 Chat History")


    if len(st.session_state.sessions) == 0:

        st.caption("No conversations yet.")

    else:

        # Show newest sessions first
        sessions_list = list(
            st.session_state.sessions.items()
        )

        sessions_list.reverse()


        for session_id, session_data in sessions_list:

            title = session_data["title"]


            # ------------------------------------------------
            # Current active chat
            # ------------------------------------------------

            if session_id == st.session_state.session_id:

                st.button(
                    f"🟢 {title}",
                    key=f"active_{session_id}",
                    use_container_width=True,
                    disabled=True
                )


            # ------------------------------------------------
            # Other chats
            # ------------------------------------------------

            else:

                if st.button(
                    f"💬 {title}",
                    key=f"history_{session_id}",
                    use_container_width=True
                ):

                    switch_chat(session_id)

                    st.rerun()


    st.markdown("---")


    # ========================================================
    # Image Configuration
    # ========================================================

    max_images = st.slider(
        "Max Images to Retrieve",
        min_value=1,
        max_value=6,
        value=3
    )


    st.markdown("---")



    # Disclaimer

    st.caption(
        "⚠️ **Disclaimer:** This tool is for "
        "educational/informational purposes only. "
        "Consult a healthcare professional for "
        "clinical diagnostics."
    )


# ============================================================
# Main Header
# ============================================================

st.markdown(
    '<div class="main-header">'
    '🩺 Skin Disease RAG Assistant'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-header">'
    'AI-Powered Diagnostic Information & Visual Reference System'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# Display Previous Conversation
# ============================================================

for message in st.session_state.messages:

    role = message["role"]
    content = message["content"]

    with st.chat_message(role):

        st.markdown(content)


        # ====================================================
        # Display Images
        # ====================================================

        if role == "assistant":

            images = message.get("images", [])


            if images:

                st.markdown(
                    "### 📸 Visual References"
                )


                cols = st.columns(
                    min(len(images), 3)
                )


                for idx, img in enumerate(images):

                    col = cols[idx % 3]


                    img_path = (
                        img.get("image_path")
                        or img.get("file_path")
                        or img.get("file_name")
                        or img.get("url")
                    )


                    disease_label = img.get(
                        "disease",
                        "Condition Reference"
                    )


                    with col:

                        st.markdown(
                            '<div class="img-card">',
                            unsafe_allow_html=True
                        )


                        # ------------------------------------------------
                        # Local image
                        # ------------------------------------------------

                        if (
                            img_path
                            and os.path.exists(img_path)
                        ):

                            st.image(
                                img_path,
                                caption=disease_label,
                                use_container_width=True
                            )


                        # ------------------------------------------------
                        # Remote image
                        # ------------------------------------------------

                        elif (
                            img_path
                            and (
                                img_path.startswith(
                                    "http://"
                                )
                                or
                                img_path.startswith(
                                    "https://"
                                )
                            )
                        ):

                            st.image(
                                img_path,
                                caption=disease_label,
                                use_container_width=True
                            )


                        # ------------------------------------------------
                        # Metadata only
                        # ------------------------------------------------

                        else:

                            st.warning(
                                f"🖼️ **Image Metadata Found**\n\n"
                                f"**Disease:** {disease_label}\n\n"
                                f"*File:* `{img_path}`"
                            )


                        st.markdown(
                            '</div>',
                            unsafe_allow_html=True
                        )


# ============================================================
# Chat Input
# ============================================================

user_query = st.chat_input(
    "Ask a question about skin diseases..."
)


# ============================================================
# Process New User Message
# ============================================================

if user_query:

    # ========================================================
    # Update Chat Title
    # ========================================================

    current_session = st.session_state.sessions[
        st.session_state.session_id
    ]


    if current_session["title"] == "New Chat":

        title = user_query.strip()

        if len(title) > 40:

            title = title[:40] + "..."

        current_session["title"] = title


    # ========================================================
    # Display User Message
    # ========================================================

    with st.chat_message("user"):

        st.markdown(user_query)


    # ========================================================
    # Store User Message
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )


    # ========================================================
    # Generate RAG Answer
    # ========================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching medical knowledge base..."
        ):

            try:

                result = generate_answer(
                    query=user_query,
                    session_id=st.session_state.session_id,
                    max_images=max_images
                )


                answer = result.get(
                    "answer",
                    "No answer was generated."
                )


                images = result.get(
                    "images",
                    []
                )


                # ====================================================
                # Display Answer
                # ====================================================

                st.markdown(
                    '<div class="answer-box">'
                    f'{answer}'
                    '</div>',
                    unsafe_allow_html=True
                )


                # ====================================================
                # Display Retrieved Images
                # ====================================================

                if images:

                    st.markdown(
                        "### 📸 Visual References"
                    )


                    cols = st.columns(
                        min(len(images), 3)
                    )


                    for idx, img in enumerate(images):

                        col = cols[idx % 3]


                        img_path = (
                            img.get("image_path")
                            or img.get("file_path")
                            or img.get("file_name")
                            or img.get("url")
                        )


                        disease_label = img.get(
                            "disease",
                            "Condition Reference"
                        )


                        with col:

                            st.markdown(
                                '<div class="img-card">',
                                unsafe_allow_html=True
                            )


                            # --------------------------------------------
                            # Local image
                            # --------------------------------------------

                            if (
                                img_path
                                and os.path.exists(img_path)
                            ):

                                st.image(
                                    img_path,
                                    caption=disease_label,
                                    use_container_width=True
                                )


                            # --------------------------------------------
                            # Remote image
                            # --------------------------------------------

                            elif (
                                img_path
                                and (
                                    img_path.startswith(
                                        "http://"
                                    )
                                    or
                                    img_path.startswith(
                                        "https://"
                                    )
                                )
                            ):

                                st.image(
                                    img_path,
                                    caption=disease_label,
                                    use_container_width=True
                                )


                            # --------------------------------------------
                            # Metadata only
                            # --------------------------------------------

                            else:

                                st.warning(
                                    f"🖼️ **Image Metadata Found**\n\n"
                                    f"**Disease:** {disease_label}\n\n"
                                    f"*File:* `{img_path}`"
                                )


                            st.markdown(
                                '</div>',
                                unsafe_allow_html=True
                            )


                # ====================================================
                # Save Assistant Message
                # ====================================================

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "images": images
                    }
                )


            except Exception as e:

                error_message = (
                    f"An error occurred while generating "
                    f"the answer: {str(e)}"
                )


                st.error(error_message)


                # Save error message
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "images": []
                    }
                )