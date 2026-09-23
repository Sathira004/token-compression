import os
import json
from typing import Any
import toons  # <-- Import the high-performance Rust-backed TOON encoder
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from mcp.types import TextContent

async def toonify_mcp_response(request: MCPToolCallRequest, handler):
    """
    Intercepts tools mid-flight. Leverages the official toons engine 
    to losslessly compress JSON payloads right before they hit the LLM context.
    """
    # 1. Execute the tool downstream to fetch raw data from the FastMCP server
    result = await handler(request)
    
    # FastMCP may return structured content or JSON inside a text content block.
    payload = getattr(result, "structuredContent", None)
    if not payload:
        text_blocks = [
            item.text
            for item in getattr(result, "content", [])
            if isinstance(item, TextContent)
        ]
        if text_blocks:
            text_payload = "\n".join(text_blocks)
            try:
                payload = json.loads(text_payload)
            except json.JSONDecodeError:
                payload = text_payload

    if payload is not None:
        # toons.dumps handles arrays of uniform objects using efficient tabular matrices.
        toon_string = toons.dumps(payload)
        
        print("\n📤 [MCP Interceptor] STEP 2: Converted TOON Payload Sent to OpenRouter LLM:")
        print("=" * 60)
        print(toon_string)
        print("=" * 60)
        
        # 4. Overwrite the primary context block traveling to the LLM
        result.content = [TextContent(type="text", text=toon_string)]
        
    return result

# 5. Instantiate your client tracking your internal subprocess servers
server_config: Any = {
    "toon_optimizer_service": {
        "transport": "stdio",
        "command": "uv",
        "args": ["run", "python", "mcp_servers/toon_server/main.py"],
    }
}

toon_mcp_client = MultiServerMCPClient(
    server_config,
    tool_interceptors=[toonify_mcp_response],
)
raw_mcp_client = MultiServerMCPClient(server_config)

async def get_integrated_mcp_tools(use_toon: bool = True):
    """Fetch MCP tools with optional TOON response interception."""
    client = toon_mcp_client if use_toon else raw_mcp_client
    return await client.get_tools()
