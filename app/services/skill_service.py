from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings


SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


class SkillSelection(BaseModel):
	skill_name: str = Field(description="The selected skill filename without the .md suffix")


def available_skills() -> list[str]:
	"""Return the skill names that are available to the selector."""
	return sorted(path.stem for path in SKILLS_DIR.glob("*.md"))


def load_skill(skill_name: str) -> str:
	"""Load a skill's system instructions from app/skills/*.md."""
	if skill_name not in available_skills():
		raise ValueError(f"Unknown skill: {skill_name}")

	skill_path = SKILLS_DIR / f"{skill_name}.md"

	if not skill_path.is_file():
		raise ValueError(f"Skill file not found: {skill_path.name}")

	return skill_path.read_text(encoding="utf-8").strip()


def select_skill(user_prompt: str) -> str:
	"""Ask the routing model to choose one of the available skill files."""
	skill_names = available_skills()
	fallback = "general_skill" if "general_skill" in skill_names else skill_names[0]

	selector = ChatOpenAI(
		model=settings.SUB_AGENT_LLM_ROUTING,
		temperature=0,
		api_key=settings.OPENROUTER_API_KEY,
		base_url="https://openrouter.ai/api/v1",
	).with_structured_output(SkillSelection)

	result = selector.invoke([
		SystemMessage(content=(
			"Choose the best skill for the user's prompt. "
			f"Return exactly one skill_name from this list: {', '.join(skill_names)}. "
			"Choose toon_plus_skill for TOON, structured data, compression, or MCP data. "
			f"Choose {fallback} when no specialized skill applies."
		)),
		HumanMessage(content=user_prompt),
	])

	selected_skill = result.skill_name.strip().removesuffix(".md")
	return selected_skill if selected_skill in skill_names else fallback
