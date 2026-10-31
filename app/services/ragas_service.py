import asyncio
import sys
import types
from datetime import datetime, timezone
from importlib import import_module
from typing import Any

from pydantic import SecretStr

from app.core.config import settings
from app.services.llm_service import create_vertex_llm


def _install_ragas_vertex_compatibility() -> None:
    """Bridge Ragas 0.4.3's removed LangChain Community Vertex import."""
    try:
        import_module("langchain_community.chat_models.vertexai")
        return
    except ModuleNotFoundError:
        from langchain_google_genai import ChatGoogleGenerativeAI
        chat_models = import_module("langchain_community.chat_models")
        llms = import_module("langchain_community.llms")

        vertex_module = types.ModuleType("langchain_community.chat_models.vertexai")
        setattr(vertex_module, "ChatVertexAI", ChatGoogleGenerativeAI)
        sys.modules[vertex_module.__name__] = vertex_module
        setattr(chat_models, "vertexai", vertex_module)
        setattr(llms, "VertexAI", ChatGoogleGenerativeAI)


async def evaluate_run(
    collection: Any,
    run_id: str,
    arm: str,
    user_input: str,
    response: str,
    retrieved_context: str,
    reference_answer: str | None = None,
) -> None:
    _install_ragas_vertex_compatibility()

    from langchain_pinecone import PineconeEmbeddings
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms import InstructorBaseRagasLLM
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithoutReference,
        ContextRecall,
        Faithfulness,
    )

    class LangChainRagasLLM(InstructorBaseRagasLLM):
        def __init__(self) -> None:
            self.model = create_vertex_llm()

        def generate(self, prompt: str, response_model: Any) -> Any:
            return self.model.with_structured_output(response_model).invoke(prompt)

        async def agenerate(self, prompt: str, response_model: Any) -> Any:
            return await self.model.with_structured_output(response_model).ainvoke(prompt)

    class PineconeRagasEmbeddings(BaseRagasEmbedding):
        def __init__(self) -> None:
            super().__init__()
            if not settings.PINECONE_EMBEDDING_MODEL or not settings.PINECONE_API_KEY:
                raise ValueError(
                    "PINECONE_EMBEDDING_MODEL and PINECONE_API_KEY are required for Ragas relevance evaluation"
                )
            self.model = PineconeEmbeddings(
                model=settings.PINECONE_EMBEDDING_MODEL,
                pinecone_api_key=SecretStr(settings.PINECONE_API_KEY),
            )

        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            return self.model.embed_query(text)

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return await asyncio.to_thread(self.model.embed_query, text)

    evaluator_llm = LangChainRagasLLM()
    evaluator_embeddings = PineconeRagasEmbeddings()

    relevancy_metric = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    relevancy_result = await relevancy_metric.ascore(
        user_input=user_input,
        response=response,
    )

    scores: dict[str, float | None] = {
        "relevance": float(relevancy_result.value),
        "answer_relevancy": float(relevancy_result.value),
        "faithfulness": None,
        "context_precision": None,
        "context_recall": None,
    }

    if retrieved_context.strip():
        precision_metric = ContextPrecisionWithoutReference(llm=evaluator_llm)
        precision_result = await precision_metric.ascore(
            user_input=user_input,
            response=response,
            retrieved_contexts=[retrieved_context],
        )
        scores["context_precision"] = float(precision_result.value)

        faithfulness_metric = Faithfulness(llm=evaluator_llm)
        faithfulness_result = await faithfulness_metric.ascore(
            user_input=user_input,
            response=response,
            retrieved_contexts=[retrieved_context],
        )
        scores["faithfulness"] = float(faithfulness_result.value)

        if reference_answer and reference_answer.strip():
            recall_metric = ContextRecall(llm=evaluator_llm)
            recall_result = await recall_metric.ascore(
                user_input=user_input,
                reference=reference_answer,
                retrieved_contexts=[retrieved_context],
            )
            scores["context_recall"] = float(recall_result.value)

    await collection.update_one(
        {"run_id": run_id},
        {
            "$set": {
                "ragas": {
                    "status": "completed",
                    "arm": arm,
                    "evaluator_model": settings.GOOGLE_LLM_MODEL_FOR_RAGAS,
                    "reference_answer": reference_answer,
                    "scores": scores,
                    "context_recall_status": (
                        "completed"
                        if scores["context_recall"] is not None
                        else "skipped_missing_reference_answer"
                    ),
                    "evaluated_at": datetime.now(timezone.utc),
                }
            }
        },
    )
