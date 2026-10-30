"""Model-directed skill selection with on-demand instruction loading."""

from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware import FilesystemMiddleware
from deepagents.middleware.filesystem import FilesystemPermission
from deepagents.profiles import GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.services.token_usage_service import call_usage


SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


class SkillSelection(BaseModel):
    """Names of skill files relevant to the current request."""

    skills: list[str] = Field(default_factory=list)


def _skill_registry() -> dict[str, tuple[Path, str]]:
    """Read only skill names/descriptions; instruction bodies remain deferred."""
    registry: dict[str, tuple[Path, str]] = {}
    for path in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        content = path.read_text(encoding="utf-8")
        parts = content.split("---", 2)
        if len(parts) != 3:
            continue
        fields: dict[str, str] = {}
        for line in parts[1].splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in {"name", "description"}:
                fields[key.strip()] = value.strip().strip("\"'")
        name, description = fields.get("name"), fields.get("description")
        if name and description:
            registry[name] = (path, description)
    return registry


def _skill_body(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    parts = content.split("---", 2)
    return parts[2].strip() if len(parts) == 3 else content.strip()

# Retained for the Deep Agents helper below; the production skill path uses one
# structured selection call and app-side loading to avoid a multi-turn discovery loop.
register_harness_profile(
    "google_genai",
    HarnessProfile(
        excluded_middleware=frozenset({"SummarizationMiddleware"}),
        excluded_tools=frozenset({"compact_conversation", "task", "ls", "glob", "grep"}),
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    ),
)


def create_skill_agent(model: BaseChatModel | None, system_prompt: str):
    """Build the Deep Agents skill interface for direct use and compatibility."""
    if model is None:
        from app.services.llm_service import create_vertex_llm

        model = create_vertex_llm()
    backend = FilesystemBackend(root_dir=str(SKILLS_DIR), virtual_mode=True)
    return create_deep_agent(
        model=model,
        backend=backend,
        # Virtual paths are relative to SKILLS_DIR, which contains skill folders.
        skills=["/"],
        middleware=[
            FilesystemMiddleware(
                backend=backend,
                tools=["read_file"],
                tool_token_limit_before_evict=None,
                human_message_token_limit_before_evict=None,
            ),
        ],
        permissions=[FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")],
        system_prompt=(
            system_prompt
            + "\n\nUse skill descriptions to decide which instructions are relevant "
            "to the user's request. Read only relevant SKILL.md files; use multiple "
            "skills for combined requests, or none when no skill applies. "
            "Read each selected skill at /<skill-name>/SKILL.md; skill names "
            "and descriptions are listed by the skills middleware. "
            "User profiles alone do not require posts-analysis. "
            "Do not delegate or modify files."
        ),
    )


async def generate_with_skills(
    model: BaseChatModel | None,
    messages: list[AnyMessage],
    system_prompt: str,
    config: RunnableConfig,
) -> tuple[AIMessage, list[dict[str, Any]], list[str]]:
    """Select from compact metadata once, then answer with selected skill bodies."""
    if model is None:
        from app.services.llm_service import create_vertex_llm

        model = create_vertex_llm()
    registry = _skill_registry()
    metadata = "\n".join(
        f"- {name}: {description}" for name, (_, description) in registry.items()
    ) or "No skills are available."
    selector_prompt = (
        "Select only the skill instructions needed to answer the user's request. "
        "Use skill descriptions as routing hints, not as evidence. Select multiple "
        "skills only when the request truly combines their domains. Return an empty "
        "list when no skill applies. Do not answer the request. Available skills:\n"
        f"{metadata}"
    )
    selection_result = await model.with_structured_output(
        SkillSelection, include_raw=True
    ).ainvoke(
        [SystemMessage(content=selector_prompt), *messages],
        config=config,
    )
    selection_message = selection_result.get("raw")
    selection_usage = call_usage(selection_message, "skill_selection")
    parsed_selection = selection_result.get("parsed")
    requested_names = parsed_selection.skills if parsed_selection else []
    selected_names = list(dict.fromkeys(name for name in requested_names if name in registry))
    loaded_skills = [str(registry[name][0]) for name in selected_names]
    selected_instructions = "\n\n".join(
        f"Instructions for {name}:\n{_skill_body(registry[name][0])}"
        for name in selected_names
    ) or "No additional skill instructions were loaded."
    response = await model.ainvoke(
        [SystemMessage(content=(
            "Apply these instructions selected for this request:\n"
            f"{selected_instructions}\n\n{system_prompt}"
        )), *messages],
        config=config,
    )
    if not isinstance(response, AIMessage) or response.tool_calls:
        raise RuntimeError("Skill agent did not produce a final assistant answer")
    return response, [selection_usage, call_usage(response, "answer")], loaded_skills
