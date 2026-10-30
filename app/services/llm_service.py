"""Shared Vertex AI chat model configuration for every application LLM call."""

from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings


def create_vertex_llm() -> ChatGoogleGenerativeAI:
    """Use the configured Gemini model through Vertex AI exclusively."""
    return ChatGoogleGenerativeAI(
        model=settings.GOOGLE_LLM_MODEL,
        api_key=settings.GOOGLE_API_KEY,
        project=settings.GOOGLE_CLOUD_PROJECT,
        location=settings.GOOGLE_CLOUD_LOCATION,
        temperature=settings.DEFAULT_MODEL_TEMPERATURE,
        vertexai=True,
    )
