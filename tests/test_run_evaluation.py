"""Check evaluation inputs without calling external evaluators or databases."""

import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage

from app.schemas.agent import ChatRequest
from app.services import run_service


class RunEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_evaluation_uses_answer_text_and_complete_original_evidence(self):
        payload = ChatRequest(message='Read Table 3 and get weather', thread_id='test', process_type='normal_rag')
        for content in ('Answer', [
            {'type': 'text', 'text': 'Answer', 'extras': {'signature': 'metadata'}},
        ]):
            with self.subTest(content=content):
                collection = AsyncMock()
                state = {
                    'messages': [AIMessage(content=content)],
                    'raw_context': 'Original table',
                    'optimized_context': 'Compressed evidence',
                    'evaluation_context': 'Original table\nWeather: 30 C',
                }
                with patch.object(run_service, 'save_run', AsyncMock(return_value='run')), patch.object(run_service, 'evaluate_run', AsyncMock()) as evaluate:
                    await run_service.persist_and_evaluate_run(collection, payload, state, 10)
                evaluate.assert_awaited_once_with(
                    collection=collection, run_id='run', arm='normal_rag',
                    user_input=payload.message, response='Answer',
                    retrieved_context='Original table\nWeather: 30 C',
                )
                collection.update_one.assert_not_awaited()
