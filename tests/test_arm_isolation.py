"""Verify input transformations and generation routing without network/model calls."""

import json
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import toons

from langchain_core.messages import AIMessage, HumanMessage

from app.agents import nodes
from app.schemas.agent import ProcessType
from app.schemas.routing import EvidencePlan
from app.services.token_usage_service import call_usage


class ArmIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_graph_collects_tools_before_encoding(self):
        from app.agents.graph import graph

        model = Mock()
        model.ainvoke = AsyncMock(return_value=AIMessage(content='Answer'))
        model.get_num_tokens.return_value = 7
        model.with_structured_output.return_value.ainvoke = AsyncMock(return_value={
            'parsed': EvidencePlan(needs_retrieval=False, tools=['mcp_get_posts']),
            'raw': AIMessage(content='Plan'),
        })
        records = [{'id': 1, 'title': 'Post'}]
        tool = SimpleNamespace(name='mcp_get_posts', ainvoke=AsyncMock(return_value=[
            {'type': 'text', 'text': json.dumps(records)},
        ]))
        with patch.object(nodes, 'get_integrated_mcp_tools', AsyncMock(return_value=[tool])), patch.object(nodes, 'create_vertex_llm', return_value=model):
            result = await graph.ainvoke(
                {'messages': [HumanMessage(content='List posts')]},
                config={'configurable': {'process_type': 'toon_only'}},
            )
        self.assertEqual(toons.loads(result['optimized_context'])['tools']['mcp_get_posts'], records)
        self.assertEqual(result['messages'][-1].content, 'Answer')
        self.assertIn(json.dumps(records), result['evaluation_context'])
        self.assertIn('tools', result['runtime_ms'])

    async def test_every_arm_uses_only_its_own_transformation(self):
        raw_document = 'Original paper evidence with number 46.3.'
        records = [{'userId': 3, 'id': 1, 'title': 'A post', 'body': 'Original body'}]
        blocks = [{'type': 'text', 'text': json.dumps(records)}]
        for mode in ProcessType:
            with self.subTest(mode=mode):
                question = HumanMessage(content='Read the paper and list posts for user 3.')
                state = {'messages': [question], 'raw_context': raw_document, 'selected_tools': ['mcp_get_posts']}
                config = {'configurable': {'process_type': mode.value}}
                tool = SimpleNamespace(name='mcp_get_posts', ainvoke=AsyncMock(return_value=blocks))
                model = Mock()
                model.get_num_tokens.return_value = 7
                response = AIMessage(content='Unmodified output', usage_metadata={
                    'input_tokens': 17, 'output_tokens': 3, 'total_tokens': 20,
                })
                model.ainvoke = AsyncMock(return_value=response)
                skill_service = ModuleType('app.services.skill_service')
                skill_service.generate_with_skills = AsyncMock(return_value=(response, [call_usage(response, 'answer')], []))
                with patch.object(nodes, 'get_integrated_mcp_tools', AsyncMock(return_value=[tool])) as fetch, patch.object(nodes, 'compress_with_llmlingua', return_value='LLMLINGUA') as lingua, patch.object(nodes.toons, 'dumps', return_value='TOON') as toon, patch.object(nodes, 'create_vertex_llm', return_value=model), patch.dict(sys.modules, {'app.services.skill_service': skill_service}):
                    state.update(await nodes.optimization_node(state, config))
                    result = await nodes.generation_node(state, config)
                fetch.assert_awaited_once_with(use_toon=False)
                self.assertEqual(state['evaluation_context'], raw_document + '\n[mcp_get_posts Results]:\n' + json.dumps(records))
                tool.ainvoke.assert_awaited_once_with({})
                if mode in (ProcessType.NORMAL_RAG, ProcessType.NAIVE_RAG):
                    lingua.assert_not_called()
                    toon.assert_not_called()
                    self.assertEqual(state['optimized_context'], raw_document + '\n[mcp_get_posts Results]:\n' + json.dumps(records))
                    self.assertIn('uncompressed document text', model.ainvoke.call_args.args[0][0]['content'])
                elif mode == ProcessType.LLMLINGUA_ONLY:
                    lingua.assert_called_once_with(raw_document, question.content)
                    toon.assert_not_called()
                    self.assertEqual(state['optimized_context'], 'LLMLINGUA' + '\n[mcp_get_posts Results]:\n' + json.dumps(records))
                else:
                    lingua.assert_not_called()
                    toon.assert_called_once_with({'documents': raw_document, 'tools': {'mcp_get_posts': records}})
                    self.assertEqual(state['optimized_context'], 'TOON')
                skill_arm = mode == ProcessType.TOON_PLUS_SKILL
                self.assertEqual(skill_service.generate_with_skills.await_count, int(skill_arm))
                self.assertEqual(model.ainvoke.await_count, int(not skill_arm))
                sent_messages = skill_service.generate_with_skills.call_args.args[1] if skill_arm else model.ainvoke.call_args.args[0][1:]
                system_content = skill_service.generate_with_skills.call_args.args[2] if skill_arm else model.ainvoke.call_args.args[0][0]['content']
                if skill_arm:
                    self.assertNotIn('Analysis instructions:', system_content)
                else:
                    for skill_name in nodes.ANALYSIS_SKILLS:
                        skill_text = (nodes.SKILLS_DIR / skill_name / 'SKILL.md').read_text(encoding='utf-8')
                        self.assertIn(f'{skill_name} instructions:\n{skill_text}', system_content)
                if mode == ProcessType.NORMAL_RAG:
                    self.assertIn('Address every requested part explicitly', system_content)
                    self.assertIn('distinguish gross reductions from net savings', system_content)
                    self.assertIn('return all available records', system_content)
                else:
                    self.assertNotIn('Address every requested part explicitly', system_content)
                self.assertEqual(sent_messages, [question])
                self.assertIs(result['messages'][0], response)
                self.assertEqual(result['token_usage'], {'input': 17, 'output': 3, 'total': 20})

    async def test_llmlingua_skips_compression_for_mcp_only_requests(self):
        records = [{'id': 1, 'body': 'Keep this natural-language field intact too.'}]
        tool = SimpleNamespace(name='mcp_get_posts', ainvoke=AsyncMock(return_value=records))
        with patch.object(nodes, 'get_integrated_mcp_tools', AsyncMock(return_value=[tool])), patch.object(nodes, 'compress_with_llmlingua') as lingua, patch.object(nodes.toons, 'dumps') as toon:
            result = await nodes.optimization_node(
                {'messages': [HumanMessage(content='List posts')], 'selected_tools': ['mcp_get_posts']},
                {'configurable': {'process_type': 'llmlingua_only'}},
            )
        lingua.assert_not_called()
        toon.assert_not_called()
        self.assertEqual(result['optimized_context'], '\n[mcp_get_posts Results]:\n' + json.dumps(records))

    def test_mcp_decoding_preserves_false_empty_and_non_json_values(self):
        for value in ([], {}, False, 0, ''):
            self.assertEqual(nodes._decode_mcp_result([{'type': 'text', 'text': json.dumps(value)}]), value)
        self.assertEqual(nodes._decode_mcp_result([{'type': 'text', 'text': 'plain text'}]), 'plain text')
        self.assertEqual(nodes._decode_mcp_result([{'userId': 1}]), [{'userId': 1}])


if __name__ == '__main__':
    unittest.main()
