import json
from retrieval import retrieve, retrieve_images
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver



llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0
)


# ============================================================
# Conversation Memory + Summarization Middleware
# ============================================================

# In-memory checkpointer.
#
# Each unique thread_id represents one conversation/session.
#
# IMPORTANT:
# This is suitable for development/testing.
# For production, we can later replace this with SQLite/PostgreSQL.
checkpointer = InMemorySaver()


summarization_middleware = SummarizationMiddleware(
    model=llm,

    # Start summarizing when conversation reaches 10 messages.
    #
    # Example:
    # User + Assistant = 2 messages
    # 5 turns = 10 messages
    trigger=("messages", 10),

    # After summarization, keep the most recent 6 messages
    # in their original form.
    keep=("messages", 6),
)


# ============================================================
# Retrieval Tool
# ============================================================

def search_skin_knowledge_base(query: str) -> str:
    """
    Search the skin disease knowledge base.

    This function is exposed to the conversational agent as a tool.
    The agent can use it to retrieve relevant documents.
    """

    results = retrieve(query, top_k=5)

    if not results:
        return json.dumps({
            "results": [],
            "images": []
        })

    # --------------------------------------------------------
    # Build text context
    # --------------------------------------------------------

    context = ""

    for result in results:
        context += f"""
Disease: {result["disease"]}
Source: {result["source_pdf"]}

{result["text"]}

"""

    # --------------------------------------------------------
    # Retrieve images from the same source PDFs
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Return both text and image metadata
    # --------------------------------------------------------

    return json.dumps({
        "context": context,
        "images": images
    })


# ============================================================
# Conversational RAG Agent
# ============================================================

SYSTEM_PROMPT = """
You are a medical information assistant specializing in skin diseases.

You answer questions using the provided skin-disease knowledge base.

IMPORTANT RULES:

1. Always use the search_skin_knowledge_base tool to retrieve
   information before answering knowledge-base questions.

2. Use the conversation history to understand follow-up questions.

3. If the user says things such as:
   - "What about its symptoms?"
   - "How is it treated?"
   - "Is it contagious?"
   - "What about the previous disease?"

   use the previous conversation to understand what they are referring to.

4. Do not invent medical information.

5. If the required information is not available in the retrieved
   knowledge base, clearly say that the information is not available
   in the provided knowledge base.

6. The retrieved context is the primary source of factual information.

7. Give clear and concise answers.

8. This system is for educational/informational purposes and is not
   a replacement for professional medical diagnosis or treatment.

When using the retrieval tool, formulate a complete search query.
For example:

Conversation:
User: What is psoriasis?
Assistant: ...

User: What are its symptoms?

The retrieval query should be something like:
"What are the symptoms of psoriasis?"

rather than simply:
"What are its symptoms?"
"""


# Create the conversational agent once.
#
# The checkpointer stores conversation state.
# The summarization middleware automatically compresses
# older conversation messages when the trigger is reached.
rag_agent = create_agent(
    model=llm,

    tools=[
        search_skin_knowledge_base
    ],

    system_prompt=SYSTEM_PROMPT,

    middleware=[
        summarization_middleware
    ],

    checkpointer=checkpointer,
)


# ============================================================
# Main RAG Function
# ============================================================

def generate_answer(
    query,
    session_id="default",
    max_images=3
):
    """
    Generate a conversational RAG answer.

    Parameters
    ----------
    query : str
        Current user question.

    session_id : str
        Unique ID for the conversation.

        Same session_id
            -> remembers previous conversation.

        Different session_id
            -> starts a separate conversation.

    max_images : int
        Maximum number of images to return.

    Returns
    -------
    dict
        {
            "answer": str,
            "images": list
        }
    """

    # --------------------------------------------------------
    # Each session gets its own LangGraph thread.
    # --------------------------------------------------------

    config = {
        "configurable": {
            "thread_id": session_id
        }
    }

    # --------------------------------------------------------
    # Invoke conversational agent
    # --------------------------------------------------------

    result = rag_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        },
        config=config
    )

    # --------------------------------------------------------
    # Get final assistant message
    # --------------------------------------------------------

    messages = result.get("messages", [])

    answer = ""

    for message in reversed(messages):

        # AIMessage
        if getattr(message, "type", None) == "ai":

            content = message.content

            if isinstance(content, str):
                answer = content

            else:
                # Handle providers returning structured content
                answer = str(content)

            break

    # --------------------------------------------------------
    # Extract retrieved images from tool messages
    # --------------------------------------------------------

    images = []

    for message in reversed(messages):

        if getattr(message, "type", None) != "tool":
            continue

        content = getattr(message, "content", "")

        try:

            tool_result = json.loads(content)

            if "images" in tool_result:

                images = tool_result["images"]

                if images:
                    break

        except (json.JSONDecodeError, TypeError):

            continue

    # --------------------------------------------------------
    # Limit images
    # --------------------------------------------------------

    images = images[:max_images]

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "answer": answer,
        "images": images
    }


# ============================================================
# Optional: Get current conversation history
# ============================================================

def get_conversation(session_id="default"):
    """
    Return the stored conversation for a session.

    Useful for debugging and later displaying chat history.
    """

    config = {
        "configurable": {
            "thread_id": session_id
        }
    }

    state = rag_agent.get_state(config)

    return state.values.get("messages", [])


# ============================================================
# Clear a conversation
# ============================================================

def clear_conversation(session_id="default"):
    """
    Clear a conversation session.

    This removes the checkpoint for the specified thread.
    """

    checkpointer.delete_thread(session_id)

