"""Routing selects actual tools rather than consuming benchmark metadata."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage, HumanMessage

from app.agents import nodes
from app.schemas.routing import EvidencePlan


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_p098_routes_to_pdf_and_users_only_in_full_graph(self):
        from app.agents.graph import graph

        question = 'Using Sections 6.1 and 6.4 of ONTO.pdf, explain the reduction gap between flat and nested datasets and the limits of the nesting-depth evidence. Also fetch user phone numbers.'
        model = Mock()
        model.get_num_tokens.return_value = 5
        model.with_structured_output.return_value.ainvoke = AsyncMock(return_value={
            'parsed': EvidencePlan(needs_retrieval=True, tools=['mcp_get_users']),
            'raw': AIMessage(content='Plan', usage_metadata={'input_tokens': 9, 'output_tokens': 1, 'total_tokens': 10}),
        })
        model.ainvoke = AsyncMock(return_value=AIMessage(
            content='Answer', usage_metadata={'input_tokens': 20, 'output_tokens': 5, 'total_tokens': 25}
        ))
        tools = [SimpleNamespace(name=name, ainvoke=AsyncMock(return_value=[{'id': 1, 'phone': '123'}]))
                 for name in ('mcp_get_users', 'mcp_get_posts', 'mcp_get_weather')]
        with patch.object(nodes, 'create_vertex_llm', return_value=model), patch.object(nodes, 'run_naive_rag', return_value='PDF evidence') as retrieve, patch.object(nodes, 'get_integrated_mcp_tools', AsyncMock(return_value=tools)):
            result = await graph.ainvoke({'messages': [HumanMessage(content=question)]}, {'configurable': {'process_type': 'toon_only'}})
        retrieve.assert_called_once_with(query=question, k=4)
        tools[0].ainvoke.assert_awaited_once_with({})
        tools[1].ainvoke.assert_not_awaited()
        tools[2].ainvoke.assert_not_awaited()
        self.assertEqual(result['selected_tools'], ['mcp_get_users'])
        self.assertEqual(result['token_usage'], {'input': 29, 'output': 6, 'total': 35})
        self.assertEqual([call['phase'] for call in result['model_calls']], ['routing', 'answer'])
        self.assertNotIn('mcp_get_posts', result['evaluation_context'])

    async def test_document_only_and_greeting_skip_mcp_connection(self):
        with patch.object(nodes, 'get_integrated_mcp_tools', AsyncMock()) as fetch:
            result = await nodes.optimization_node(
                {'selected_tools': [], 'raw_context': 'Paper'},
                {'configurable': {'process_type': 'normal_rag'}},
            )
        fetch.assert_not_awaited()
        self.assertEqual(result['evaluation_context'], 'Paper')
        with patch.object(nodes, 'create_vertex_llm') as create:
            plan = await nodes.route_query({'messages': [HumanMessage(content='hi')]})
        create.assert_not_called()
        self.assertEqual(plan['selected_tools'], [])
        self.assertFalse(plan['needs_retrieval'])

    async def test_missing_requested_tool_is_reported(self):
        with patch.object(nodes, 'get_integrated_mcp_tools', AsyncMock(return_value=[])):
            result = await nodes.optimization_node(
                {'selected_tools': ['mcp_get_weather']}, {'configurable': {'process_type': 'normal_rag'}}
            )
        self.assertIn('Requested tool is unavailable', result['evaluation_context'])

    async def test_invalid_plan_does_not_silently_fetch_all_tools(self):
        model = Mock()
        model.with_structured_output.return_value.ainvoke = AsyncMock(return_value={'parsed': None, 'parsing_error': ValueError('bad')})
        with patch.object(nodes, 'create_vertex_llm', return_value=model):
            with self.assertRaisesRegex(ValueError, 'routing plan'):
                await nodes.route_query({'messages': [HumanMessage(content='Get weather')]})
