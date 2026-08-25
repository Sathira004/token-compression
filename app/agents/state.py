import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    # Appends new messages automatically instead of over-writing them
    messages: Annotated[list[AnyMessage], operator.add] 
    
    # Keeps track of the raw text context fetched from your engines
    raw_context: str
    
    # Keeps track of text optimized by LLMLingua or TOON
    optimized_context: str
