import operator
from typing import Any
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage

class TokenUsage(TypedDict):
    input: int
    output: int
    total: int

class RuntimeMetrics(TypedDict, total=False):
    routing: float
    token_counting: float
    retrieval: float
    tools: float
    compression: float
    generation: float

class AgentState(TypedDict):
    needs_retrieval: bool
    selected_tools: list[str]
    model_calls: list[dict[str, Any]]
    loaded_skills: list[str]
    context_tokens: dict[str, Any]
    # Appends new messages automatically instead of over-writing them
    messages: Annotated[list[AnyMessage], operator.add] 
    
    # Keeps track of the raw text context fetched from your engines
    raw_context: str
    
    # Keeps track of text optimized by LLMLingua or TOON
    optimized_context: str

    # Original document and tool evidence, shared across evaluation arms
    evaluation_context: str

    # Total usage across routing, on-demand skill loading, and answering
    token_usage: TokenUsage

    # Node timings measured in milliseconds
    runtime_ms: RuntimeMetrics
