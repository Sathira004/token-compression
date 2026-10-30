from threading import Lock
from typing import Any

from app.core.config import settings


_compressor: Any | None = None
_compressor_lock = Lock()


def _get_compressor() -> Any:
	"""Create the model only when LLMLingua compression is first requested."""
	global _compressor

	if _compressor is None:
		with _compressor_lock:
			if _compressor is None:
				from llmlingua import PromptCompressor

				_compressor = PromptCompressor(
					model_name=settings.LLMLINGUA_MODEL_NAME,
					device_map=settings.LLMLINGUA_DEVICE,
					use_llmlingua2=settings.LLMLINGUA_USE_LLMLINGUA2,
				)

	return _compressor


def compress_with_llmlingua(context: str, user_prompt: str) -> str:
	"""Compress retrieved document prose; use the unchanged user question as guidance."""
	context_parts = [
		part.strip() for part in context.split("\n\n") if part and part.strip()
	]
	if not context_parts:
		return ""

	result = _get_compressor().compress_prompt(
		context_parts,
		question=user_prompt,
		rate=settings.LLMLINGUA_RATE,
	)
 
	print("\n📤 [Optimization Node] LLMLingua-Compressed Context:")
	print(result["compressed_prompt"])
 
	return str(result["compressed_prompt"])
