from typing_extensions import TypedDict
from typing import List, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState
from enum import StrEnum


class StateKeys(StrEnum):
    MESSAGES = "messages"
    REMAINING_CYCLES = "remaining_cycles"
    VERIFICATION_SCORE = "verification_score"
    NEXT_AGENT = "next_agent"
    REASONING = "reasoning"
    VISITED = "visited"
    HOPS_LEFT = "hops_left"


class PanelState(MessagesState):
    visited: List[str]
    hops_left: int
    next_agent: str | None


class AgentState(MessagesState):
    next_agent: str | None
    reasoning: str | None


class ToolState(MessagesState):
    remaining_cycles: int
    verification_score: float
