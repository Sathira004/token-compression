import os
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
    
    # 2. Extract structuredContent JSON data blocks if published by the server
    if hasattr(result, "structuredContent") and result.structuredContent:
        
        # 3. Use the official encoder to compress JSON to token-optimized notation
        # toons.dumps handles arrays of uniform objects using efficient tabular matrices
        toon_string = toons.dumps(result.structuredContent)
        
        print("\n📤 [MCP Interceptor] STEP 2: Converted TOON Payload Sent to OpenRouter LLM:")
        print("=" * 60)
        print(toon_string)
        print("=" * 60)
        
        # 4. Overwrite the primary context block traveling to the LLM
        result.content = [TextContent(type="text", text=toon_string)]
        
    return result

# 5. Instantiate your client tracking your internal subprocess servers
mcp_client = MultiServerMCPClient(
    {
        "toon_optimizer_service": {
            "transport": "stdio",
            "command": "uv",
            "args": ["run", "python", "mcp_servers/toon_server/main.py"]
        }
    },
    tool_interceptors=[toonify_mcp_response] # Enforce automated encoding onto all tools
)

async def get_integrated_mcp_tools():
    """Fetches tools dynamically wrapped in LangChain execution structures."""
    return await mcp_client.get_tools()
