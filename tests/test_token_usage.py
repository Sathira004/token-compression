"""Keep per-call totals auditable and unavailable measurements distinct from zero."""

import unittest
from unittest.mock import AsyncMock, Mock

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app.schemas.agent import ChatRequest
from app.services.run_service import save_run
from app.services.token_usage_service import ModelCallRecorder, call_usage, measure_context, sum_usage


class TokenUsageTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_counts_use_provider_and_reuse_identical_context(self):
        model = Mock()
        model.get_num_tokens.side_effect = lambda text: {'raw': 30, 'toon': 20}[text]
        result = await measure_context(model, 'raw', 'toon')
        self.assertEqual((result['raw'], result['after_compression']), (30, 20))
        self.assertEqual(result['method'], 'provider_count_tokens')
        model.get_num_tokens.reset_mock()
        result = await measure_context(model, 'raw', 'raw')
        model.get_num_tokens.assert_called_once_with('raw')
        self.assertEqual((result['raw'], result['after_compression']), (30, 30))

    async def test_counting_failure_is_null_but_empty_context_is_zero(self):
        model = Mock()
        model.get_num_tokens.side_effect = RuntimeError('unavailable')
        result = await measure_context(model, 'data', 'toon')
        self.assertEqual(result['status'], 'unavailable')
        self.assertIsNone(result['raw'])
        self.assertIsNone(result['after_compression'])
        model.get_num_tokens.reset_mock()
        empty = await measure_context(model, '', '')
        model.get_num_tokens.assert_not_called()
        self.assertEqual((empty['raw'], empty['after_compression']), (0, 0))

    def test_recorder_counts_calls_without_model_name_and_marks_missing_usage(self):
        recorder = ModelCallRecorder('skill_selection')
        for message in (
            AIMessage(content='read', usage_metadata={'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12}),
            AIMessage(content='ready', usage_metadata={'input_tokens': 15, 'output_tokens': 1, 'total_tokens': 16}),
            AIMessage(content='no usage'),
        ):
            recorder.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
        self.assertEqual(len(recorder.calls), 3)
        self.assertEqual(sum_usage(recorder.calls), {'input': 25, 'output': 3, 'total': 28})
        self.assertFalse(recorder.calls[-1]['usage_available'])

    async def test_persisted_totals_include_routing_and_selection_with_context_counts(self):
        calls = [call_usage(AIMessage(content='test', usage_metadata={
            'input_tokens': amount, 'output_tokens': 2, 'total_tokens': amount + 2,
        }), phase) for phase, amount in [('routing', 10), ('skill_selection', 20), ('answer', 100)]]
        collection = AsyncMock()
        payload = ChatRequest(message='Question', thread_id='test', process_type='toon_plus_skill')
        await save_run(collection, payload, {
            'token_usage': sum_usage(calls), 'model_calls': calls,
            'context_tokens': {'raw': 100, 'after_compression': 70, 'status': 'completed'},
            'selected_tools': ['mcp_get_users'], 'loaded_skills': ['/vector-db-analysis/SKILL.md'],
        }, 20)
        document = collection.insert_one.call_args.args[0]
        tokens = document['tokens']
        self.assertEqual(tokens['total'], 136)
        self.assertEqual(tokens['input'], 130)
        self.assertEqual(tokens['context_raw'], 100)
        self.assertEqual(tokens['context_after_compression'], 70)
        self.assertEqual(tokens['by_phase']['skill_selection']['total'], 22)
        self.assertEqual(tokens['calls'], calls)
        self.assertEqual(document['selected_tools'], ['mcp_get_users'])
        self.assertEqual(document['loaded_skills'], ['/vector-db-analysis/SKILL.md'])
