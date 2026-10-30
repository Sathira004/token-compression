from datetime import datetime, timezone
from typing import Any

from pymongo.asynchronous.collection import AsyncCollection

from app.core.config import settings
from app.schemas.agent import ChatRequest
from app.services.ragas_service import evaluate_run
from app.services.token_usage_service import phase_usage


def get_run_id(payload: ChatRequest) -> str:
    return payload.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


async def save_run(
    collection: AsyncCollection,
    payload: ChatRequest,
    final_state: dict[str, Any],
    end_to_end_ms: float,
) -> str:
    token_usage = final_state.get("token_usage", {})
    runtime_ms = final_state.get("runtime_ms", {})
    run_id = get_run_id(payload)
    context_tokens = final_state.get("context_tokens", {})
    calls = final_state.get("model_calls", [])

    await collection.insert_one(
        {
            "run_id": run_id,
            "prompt_id": payload.prompt_id or payload.thread_id,
            "task_type": payload.task_type or payload.process_type.value,
            "arm": payload.arm or payload.process_type.value,
            "repeat": payload.repeat,
            "model": payload.model or settings.GOOGLE_LLM_MODEL,
            "tokens": {
                "context_raw": context_tokens.get("raw"),
                "context_after_compression": context_tokens.get("after_compression"),
                "context_measurement": context_tokens,
                "measurement_version": 2,
                "calls": calls,
                "by_phase": phase_usage(calls),
                "input": int(token_usage.get("input", 0)),
                "output": int(token_usage.get("output", 0)),
                "prompt_total": int(token_usage.get("input", 0)),
                "completion": int(token_usage.get("output", 0)),
                "total": int(token_usage.get("total", 0)),
            },
            "latency_ms": {
                "routing": round(float(runtime_ms.get("routing", 0)), 2),
                "token_counting": round(float(runtime_ms.get("token_counting", 0)), 2),
                "compression": round(float(runtime_ms.get("compression", 0)), 2),
                "retrieval": round(float(runtime_ms.get("retrieval", 0)), 2),
                "tools": round(float(runtime_ms.get("tools", 0)), 2),
                "generation": round(float(runtime_ms.get("generation", 0)), 2),
                "end_to_end": round(end_to_end_ms, 2),
            },
            "ragas": {
                "status": "pending",
                "arm": payload.arm or payload.process_type.value,
                "input": payload.ragas,
            },
            "selected_tools": final_state.get("selected_tools", []),
            "loaded_skills": final_state.get("loaded_skills", []),
            "created_at": datetime.now(timezone.utc),
        }
    )
    return run_id


async def persist_and_evaluate_run(
    collection: AsyncCollection,
    payload: ChatRequest,
    final_state: dict[str, Any],
    end_to_end_ms: float,
) -> None:
    run_id = await save_run(collection, payload, final_state, end_to_end_ms)
    try:
        response = final_state.get("messages", [])[-1].text
        context = final_state.get("evaluation_context", final_state.get("raw_context", ""))
        await evaluate_run(
            collection=collection,
            run_id=run_id,
            arm=payload.arm or payload.process_type.value,
            user_input=payload.message,
            response=response,
            retrieved_context=str(context),
        )
    except Exception as exc:
        await collection.update_one(
            {"run_id": run_id},
            {
                "$set": {
                    "ragas": {
                        "status": "failed",
                        "arm": payload.arm or payload.process_type.value,
                        "error": str(exc),
                        "evaluated_at": datetime.now(timezone.utc),
                    }
                }
            },
        )
