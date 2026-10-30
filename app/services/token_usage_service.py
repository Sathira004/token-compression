"""Provider-reported call usage and separate evidence-size measurements."""

import asyncio
from threading import Lock
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult


def call_usage(message: Any, phase: str) -> dict[str, Any]:
    metadata = getattr(message, "response_metadata", {}) or {}
    usage = getattr(message, "usage_metadata", None) or metadata.get("token_usage", {})
    input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    return {
        "phase": phase,
        "model": metadata.get("model_name") or metadata.get("model_version"),
        "input": input_tokens,
        "output": output_tokens,
        "total": int(usage.get("total_tokens", input_tokens + output_tokens) or 0),
        "usage_available": bool(usage),
        "tools_requested": [call["name"] for call in getattr(message, "tool_calls", [])],
    }


def sum_usage(calls: list[dict[str, Any]]) -> dict[str, int]:
    return {key: sum(call[key] for call in calls) for key in ("input", "output", "total")}


def phase_usage(calls: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        phase: sum_usage([call for call in calls if call["phase"] == phase])
        for phase in dict.fromkeys(call["phase"] for call in calls)
    }


class ModelCallRecorder(BaseCallbackHandler):
    """Capture each discovery call, even when model-name metadata is absent."""

    def __init__(self, phase: str):
        self.phase = phase
        self.calls: list[dict[str, Any]] = []
        self._lock = Lock()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if not response.generations or not response.generations[0]:
            return
        generation = response.generations[0][0]
        if isinstance(generation, ChatGeneration) and isinstance(generation.message, AIMessage):
            with self._lock:
                self.calls.append(call_usage(generation.message, self.phase))


async def measure_context(model: Any, raw: str, optimized: str) -> dict[str, Any]:
    """Count evidence with the provider tokenizer, never substitute fake zeroes."""
    async def count(text: str) -> int:
        return await asyncio.to_thread(model.get_num_tokens, text) if text else 0

    try:
        if raw == optimized:
            raw_count = optimized_count = await count(raw)
        else:
            raw_count, optimized_count = await asyncio.gather(count(raw), count(optimized))
        return {
            "raw": raw_count, "after_compression": optimized_count,
            "method": "provider_count_tokens", "status": "completed",
        }
    except Exception:
        # Token counting must not prevent answering. Unavailable is not zero.
        return {
            "raw": None, "after_compression": None,
            "method": "provider_count_tokens", "status": "unavailable",
        }
