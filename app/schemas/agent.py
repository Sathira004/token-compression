from enum import Enum
from pydantic import BaseModel, Field

class ProcessType(str, Enum):
    NORMAL_RAG = "normal_rag"
    NAIVE_RAG = "naive_rag"
    LLMLINGUA_ONLY = "llmlingua_only"
    TOON_ONLY = "toon_only"
    TOON_SKILL = "toon_skill"
    TOON_PLUS_SKILL = "toon_plus_skill"

class ChatRequest(BaseModel):
    message: str = Field(..., description="The prompt or query from the user UI")
    thread_id: str = Field(..., description="Unique ID to maintain user chat history memory context")
    process_type: ProcessType = Field(..., description="The chosen active toggle setting from the frontend dashboard")
    run_id: str | None = None
    prompt_id: str | None = None
    task_type: str | None = None
    arm: str | None = None
    repeat: int = Field(default=1, ge=1)
    model: str | None = None
    ragas: dict[str, float] = Field(default_factory=dict)
    reference_answer: str | None = Field(
        default=None,
        description="Expected answer used to calculate reference-based context recall",
    )
