from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Project Info
    PROJECT_NAME: str = "Token Compression API"
    
    # API Keys (Will throw validation error if completely missing when requested)
    OPENROUTER_API_KEY: str
    
    # Model Controls
    DEFAULT_LLM_MODEL: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    DEFAULT_MODEL_TEMPERATURE: float = 0.0
    SUB_AGENT_LLM_ROUTING: str

    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str
    PINECONE_EMBEDDING_MODEL: str 

    # Automatically extract key configurations directly out of a root level .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Instantiate a global settings variable accessible to any node module
settings = Settings()