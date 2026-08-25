from pydantic import BaseModel, Field
from app.schemas.agent import ProcessType
from pydantic_settings import BaseSettings, SettingsConfigDict

class GraphConfig(BaseModel):
    # Configurable runtime flag tracking your 4 switch options
    process_type: ProcessType = Field(default=ProcessType.NAIVE_RAG)
