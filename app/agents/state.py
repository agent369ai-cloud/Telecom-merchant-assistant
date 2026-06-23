from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class MerchantAgentState(TypedDict):
    """Immutable state passed between agent nodes."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    merchant_id: str
    current_tier: str
    next_node: str
    is_compliant: bool