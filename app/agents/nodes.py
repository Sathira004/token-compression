from typing import Dict, Any
import toons  # <-- High-performance Rust-backed TOON encoder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from app.agents.state import AgentState
from app.agents.config import GraphConfig
from app.core.config import settings
from app.services.rag_service import run_naive_rag
from app.mcp.client import get_integrated_mcp_tools



def route_query(state: AgentState) -> str:
    """Decides whether the user's query requires vector search retrieval."""
    user_message = state["messages"][-1].content if state.get("messages") else ""
    
    # Fast heuristic for trivial greetings to bypass LLM classification latency
    if user_message.lower().strip() in ["hi", "hello", "hey", "who are you?", "thanks", "bye"]:
        return "generate_response"
    
    router_llm = ChatOpenAI(
        model=settings.SUB_AGENT_LLM_ROUTING,
        temperature=0.0,
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    
    classification = router_llm.invoke([
        SystemMessage(content=(
            "You are a query classifier. Output 'RETRIEVE' if the question requires external "
            "document/domain knowledge lookup, otherwise output 'DIRECT'."
        )),
        HumanMessage(content=user_message)
    ])
    
    decision = classification.content.strip().upper()
    if "RETRIEVE" in decision:
        return "retrieve_documents"
    return "generate_response"


def retrieval_node(state: AgentState) -> Dict[str, Any]:
    """Retrieves raw context documents from Pinecone for the user's query."""
    print("\n\n[Retrieval Node] Querying Pinecone...")
    
    user_message = ""
    if state.get("messages"):
        last_msg = state["messages"][-1]
        user_message = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    raw_context = ""
    if user_message:
        # Pull records out of your Integrated Inference API setup
        raw_context = run_naive_rag(query=user_message, k=4)

    return {"raw_context": raw_context}


def optimization_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Evaluates the 5-way process type switch requested by the frontend.
    Leverages the official 'toons' library to natively encode context matrix streams.
    """
    print("\n\n[Optimization Node] Processing context switch...")
    
    # 1. Safely extract the frontend configuration parameters mapping
    graph_config = GraphConfig.model_validate(config.get("configurable", {}))
    mode = graph_config.process_type
    
    raw_context = state.get("raw_context", "")
    
    # 2. Process Switch Routing 
    if mode == "normal_rag" or mode == "naive_rag":
        # Pass raw context string directly 1:1 with zero optimization changes
        print(f" -> Mode: '{mode}'. Bypassing token optimization layers.")
        optimized_context = raw_context
        
    # elif mode == "llmlingua_only":
    #     print(" -> Mode: 'llmlingua_only'. Running LLMLingua compression pipeline.")
    #     optimized_context = compress_with_lingua(raw_context)
        
    elif mode == "toon_only":
        print(" -> Mode: 'toon_only'. Compressing structure with official 'toons' library.")
        # If your raw context is a string, wrap it or parse it if it is JSON structured
        # toons.dumps handles serialization into Token-Oriented Object Notation perfectly
        optimized_context = toons.dumps(raw_context)
        
        print("\n📤 [Optimization Node] TOON-Encoded Context:")
        print(optimized_context)
        
    # elif mode == "toon_skill":
    #     print(" -> Mode: 'toon_skill'. Extracting dynamic skills and encoding to TOON format.")
    #     skills = load_skills(state["messages"])
    #     # Format context alongside your instruction sets using TOON syntax rules
    #     combined_payload = {"context": raw_context, "loaded_skills": skills}
    #     optimized_context = toons.dumps(combined_payload)
        
    else:
        optimized_context = raw_context

    return {"optimized_context": optimized_context}


async def generation_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates the final response via OpenRouter. 
    Executes MCP tools to fetch ToonServer data and includes it in the context.
    """
    print("\n\n[Generation Node] Compiling model context with MCP integration...")
    
    # 1. Fetch live tool vectors configured with our Option A TOON interceptor
    mcp_tools = await get_integrated_mcp_tools()
    
    # 2. Execute MCP tools to fetch ToonServer data
    toon_data_context = ""
    if mcp_tools:
        for tool in mcp_tools:
            try:
                print(f"  → Executing MCP tool: {tool.name}")
                # Call the tool to fetch ToonServer data
                tool_result = await tool.ainvoke({"category": "token_compression"})
                
                # Serialize tool results to TOON format
                import toons
                toon_data_context += f"\n[{tool.name} Results]:\n"
                toon_data_context += toons.dumps(tool_result)
                print(f"  ✓ ToonServer data retrieved and compressed\n")
            except Exception as e:
                print(f"  ✗ Error executing tool {tool.name}: {str(e)}\n")
    
    # 3. Build OpenRouter model configuration
    llm = ChatOpenAI(
        model=settings.DEFAULT_LLM_MODEL, 
        temperature=settings.DEFAULT_MODEL_TEMPERATURE,
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # 4. Extract the active optimized context (standard text, LLMLingua, or TOON strings)
    context_to_use = state.get("optimized_context", "")
    
    # 5. Combine all context sources: RAG context + ToonServer data
    full_context = context_to_use + toon_data_context
    
    system_prompt = {
        "role": "system", 
        "content": (
            "You are a highly efficient assistant operating within a strict context pipeline. "
            "Use the provided context to answer the user request. Context includes token data "
            f"in compressed Token-Oriented Object Notation (TOON) layout:\n\n{full_context}"
        )
    }
    
    # 6. Invoke OpenRouter model with full context
    response = await llm.ainvoke([system_prompt] + state["messages"])

    print(f"\n\n[Generation Node] Response Generated: {response.content[:100]}...\n\n")

    return {"messages": [response]}
