"""Exercise real Deep Agents skill middleware with a scripted model (no API calls)."""

import unittest
import sys
from types import ModuleType
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import Field

from app.services.skill_service import create_skill_agent, generate_with_skills
from app.services.token_usage_service import sum_usage


class ScriptedModel(FakeMessagesListChatModel):
    seen: list = Field(default_factory=list)
    tool_names: set[str] = Field(default_factory=set)

    def _get_ls_params(self, **kwargs):
        # Exercise the same harness profile as the production Vertex AI model.
        return {**super()._get_ls_params(**kwargs), "ls_provider": "google_genai"}

    def bind_tools(self, tools, **kwargs):
        self.tool_names.update(t.name if hasattr(t, "name") else t["name"] for t in tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def answer(content="Done", tool_calls=None):
    return AIMessage(
        content=content,
        tool_calls=tool_calls or [],
        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        response_metadata={"model_name": "scripted-model"},
    )


def read_skill(name, call_id):
    return {"name": "read_file", "args": {"file_path": f"/{name}/SKILL.md"}, "id": call_id}


class SkillIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_model_comes_from_vertex_factory(self):
        model = ScriptedModel(responses=[answer('READY'), answer()])
        service = ModuleType("app.services.llm_service")
        service.create_vertex_llm = Mock(return_value=model)
        with patch.dict(sys.modules, {"app.services.llm_service": service}):
            response, calls, loaded = await generate_with_skills(
                None, [HumanMessage(content="Hello")], "Answer the user.", {}
            )
        service.create_vertex_llm.assert_called_once_with()
        self.assertEqual(response.content, "Done")
        self.assertEqual(sum_usage(calls)["total"], 24)
        self.assertEqual(loaded, [])

    async def test_no_skill_reads_for_direct_answer(self):
        model = ScriptedModel(responses=[answer("READY"), answer("Hello")])
        response, calls, loaded = await generate_with_skills(model, [HumanMessage(content="Hello")], "Answer the user.", {})
        self.assertEqual(response.content, "Hello")
        self.assertEqual(sum_usage(calls), {"input": 20, "output": 4, "total": 24})
        self.assertEqual(len(model.seen), 2)
        self.assertEqual(loaded, [])
        self.assertEqual([call['phase'] for call in calls], ['skill_selection', 'answer'])
        prompt = str(model.seen[0][0].content)
        for name in ("weather-analysis", "posts-analysis", "vector-db-analysis"):
            self.assertIn(name, prompt)
        # Metadata is visible, but the instruction bodies have not been injected.
        self.assertNotIn("Do not count a TOON header as a record", prompt)
        self.assertNotIn("Do not assume units when they are absent", prompt)
        self.assertNotIn("A similarity-search result is a partial selection", prompt)
        self.assertIn("read_file", model.tool_names)
        self.assertTrue({"ls", "glob", "grep"}.isdisjoint(model.tool_names))
        self.assertTrue({"write_file", "edit_file", "delete", "execute"}.isdisjoint(model.tool_names))
        self.assertTrue({"task", "compact_conversation"}.isdisjoint(model.tool_names))

    async def test_skill_graph_has_no_automatic_summarization(self):
        agent = create_skill_agent(ScriptedModel(responses=[answer()]), "Answer the user.")
        self.assertFalse(any("summar" in name.lower() for name in agent.get_graph().nodes))

    async def test_weather_body_read_on_demand_and_all_calls_counted(self):
        model = ScriptedModel(responses=[
            answer("", [read_skill("weather-analysis", "weather")]),
            answer("READY"),
            answer("Temperature is unavailable."),
        ])
        response, calls, loaded = await generate_with_skills(
            model, [HumanMessage(content="What is the weather?")], "Use supplied context. EVIDENCE_SENTINEL", {}
        )
        tool_messages = [m for m in model.seen[1] if isinstance(m, ToolMessage)]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("current_units", tool_messages[0].content)
        self.assertEqual(tool_messages[0].status, "success")
        self.assertEqual(response.content, "Temperature is unavailable.")
        self.assertEqual(sum_usage(calls), {"input": 30, "output": 6, "total": 36})
        self.assertEqual(loaded, ['/weather-analysis/SKILL.md'])
        for request in model.seen[:-1]:
            self.assertNotIn('EVIDENCE_SENTINEL', str(request))
        final_prompt = str(model.seen[-1][0].content)
        self.assertIn('EVIDENCE_SENTINEL', final_prompt)
        self.assertIn('Do not assume units when they are absent', final_prompt)
        self.assertNotIn('Do not count a TOON header as a record', final_prompt)
        self.assertNotIn('A similarity-search result is a partial selection', final_prompt)
        self.assertEqual([call['phase'] for call in calls], ['skill_selection', 'skill_selection', 'answer'])

    async def test_multiple_selected_skills_reach_answer_without_discovery_output(self):
        model = ScriptedModel(responses=[
            answer('', [read_skill('posts-analysis', 'posts'), read_skill('vector-db-analysis', 'paper')]),
            answer('DO_NOT_TRANSFER_THIS_SPECULATION'),
            answer('Final answer'),
        ])
        response, calls, loaded = await generate_with_skills(
            model, [HumanMessage(content='Analyze the paper and posts')], 'LARGE_TOON_EVIDENCE', {}
        )
        self.assertEqual(response.content, 'Final answer')
        self.assertEqual(len(calls), 3)
        self.assertEqual(set(loaded), {'/posts-analysis/SKILL.md', '/vector-db-analysis/SKILL.md'})
        final_prompt = str(model.seen[-1][0].content)
        self.assertIn('post.userId = user.id', final_prompt)
        self.assertIn('A similarity-search result is a partial selection', final_prompt)
        self.assertNotIn('DO_NOT_TRANSFER_THIS_SPECULATION', final_prompt)
        self.assertNotIn('LARGE_TOON_EVIDENCE', str(model.seen[:-1]))

    async def test_multiple_skills_for_combined_request(self):
        model = ScriptedModel(responses=[
            answer("", [read_skill("posts-analysis", "posts"), read_skill("vector-db-analysis", "paper")]),
            answer(),
        ])
        result = await create_skill_agent(model, "Analyze the supplied evidence.").ainvoke(
            {"messages": [HumanMessage(content="Summarize ONTO.pdf and count posts.")]}
        )
        bodies = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(len(bodies), 2)
        self.assertTrue(all(m.status == "success" for m in bodies))
        self.assertIn("post.userId = user.id", str(bodies))
        self.assertIn("Object Notation for Token Optimization", str(bodies))

    async def test_backend_rejects_parent_directory_access(self):
        model = ScriptedModel(responses=[
            answer("", [{"name": "read_file", "args": {"file_path": "../core/config.py"}, "id": "outside"}]),
            answer(),
        ])
        result = await create_skill_agent(model, "Answer the user.").ainvoke(
            {"messages": [HumanMessage(content="Read a file.")]}
        )
        tools = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].status, "error")


if __name__ == "__main__":
    unittest.main()
