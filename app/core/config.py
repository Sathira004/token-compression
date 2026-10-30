from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

class Settings(BaseSettings):
    # Project Info
    PROJECT_NAME: str = "Token Compression API"
    
    # Model Controls
    DEFAULT_MODEL_TEMPERATURE: float = 0.0

    # LLMLingua compression controls
    LLMLINGUA_MODEL_NAME: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
    LLMLINGUA_DEVICE: str = "cpu"
    LLMLINGUA_RATE: float = 0.5
    LLMLINGUA_USE_LLMLINGUA2: bool = True

    # Pinecone
    PINECONE_API_KEY: str | None = None
    PINECONE_INDEX_NAME: str | None = None
    PINECONE_EMBEDDING_MODEL: str | None = None 
    
    # Google Gemini / Vertex AI
    GOOGLE_API_KEY: str | None = None
    GOOGLE_CLOUD_PROJECT: str = ""
    GOOGLE_CLOUD_LOCATION: str = "global"
    GOOGLE_LLM_MODEL: str = "gemini-3.1-flash-lite"
    GOOGLE_LLM_MODEL_FOR_RAGAS: str = "gemini-2.5-flash"

    # MongoDB
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_USERNAME: str | None = None
    MONGODB_PASSWORD: str | None = None
    MONGODB_DATABASE: str = "token_compression"
    MONGODB_RUNS_COLLECTION: str = "runs"

    # Automatically extract key configurations directly out of a root level .env file
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        # Ignore obsolete provider keys in existing .env files.
        extra="ignore",
    )


# Instantiate a global settings variable accessible to any node module
settings = Settings()
if not settings.GOOGLE_CLOUD_PROJECT:
    raise ValueError("GOOGLE_CLOUD_PROJECT must be set for Vertex AI")
