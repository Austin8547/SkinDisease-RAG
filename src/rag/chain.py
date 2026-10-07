import json

from .agent import rag_agent
from .memory import checkpointer


def generate_answer(
    query,
    session_id="default",
    max_images=3
):
    config = {
        "configurable": {
            "thread_id": session_id
        }
    }

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

    messages = result.get("messages", [])

    answer = ""

    for message in reversed(messages):

        if getattr(message, "type", None) == "ai":

            content = message.content

            if isinstance(content, str):
                answer = content
            else:
                answer = str(content)

            break

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

    images = images[:max_images]

    return {
        "answer": answer,
        "images": images
    }


def get_conversation(session_id="default"):

    config = {
        "configurable": {
            "thread_id": session_id
        }
    }

    state = rag_agent.get_state(config)

    return state.values.get("messages", [])


def clear_conversation(session_id="default"):

    checkpointer.delete_thread(session_id)