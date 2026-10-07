from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from .llm import llm


checkpointer = InMemorySaver()


summarization_middleware = SummarizationMiddleware(
    model=llm,
    trigger=("messages", 10),
    keep=("messages", 6),
)