from langchain.agents import create_agent

from .llm import llm
from .memory import (
    checkpointer,
    summarization_middleware
)
from .tools import search_skin_knowledge_base
from .prompt import SYSTEM_PROMPT


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