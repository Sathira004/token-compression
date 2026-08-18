from enum import Enum
from pydantic import BaseModel, Field

class ProcessType(str, Enum):
    NORMAL_RAG = "normal_rag"
    NAIVE_RAG = "naive_rag"
    LLMLINGUA_ONLY = "llmlingua_only"
    TOON_ONLY = "toon_only"
    TOON_SKILL = "toon_skill"

class ChatRequest(BaseModel):
    message: str = Field(..., description="The prompt or query from the user UI")
    thread_id: str = Field(..., description="Unique ID to maintain user chat history memory context")
    process_type: ProcessType = Field(..., description="The chosen active toggle setting from the frontend dashboard")
