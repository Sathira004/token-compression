from langchain_core.tools import tool

@tool
def request_mcp_pipeline_details(metric_type: str) -> str:
    """Fetches localized structural optimization context directly via MCP."""
    # Logic to request context from your mcp_servers/ folder
    return f"MCP successfully fetched metadata for {metric_type}"

# Export your tools list to bind into your LLM call step
tools = [request_mcp_pipeline_details]
