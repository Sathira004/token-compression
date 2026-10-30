"""One model-generated evidence plan, shared by every experimental arm."""

from typing import Literal

from pydantic import BaseModel, Field


class EvidencePlan(BaseModel):
    needs_retrieval: bool = Field(description="Whether the request needs indexed document evidence")
    tools: list[Literal["mcp_get_users", "mcp_get_posts", "mcp_get_weather"]] = Field(
        description="Only the MCP tools needed to answer this request; empty for document-only or general questions"
    )
