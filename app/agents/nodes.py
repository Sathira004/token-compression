from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from app.agents.state import AgentState
from app.core.config import settings  # <-- Import your configuration class


def retrieval_node(state: AgentState) -> Dict[str, Any]:
    """Retrieves raw context documents for the user's query.
    
    This is a placeholder — replace with your actual retrieval logic
    (e.g. vector store lookup, API call, database query).
    """
    # TODO: Implement actual retrieval logic
    user_message = state["messages"][-1].content if state["messages"] else ""
    raw_context = f"Retrieved context for: {user_message}"
    return {"raw_context": raw_context}


def optimization_node(state: AgentState) -> Dict[str, Any]:
    """Optimizes/compresses the raw context using token compression.
    
    This is a placeholder — replace with LLMLingua, TOON, or your
    preferred compression strategy.
    """
    # TODO: Implement actual token compression logic
    raw = state.get("raw_context", "")
    optimized_context = raw  # Pass-through for now
    return {"optimized_context": optimized_context}


def generation_node(state: AgentState) -> Dict[str, Any]:
    """Generates the final response using configuration-driven parameters."""
    
    llm = ChatOpenAI(
        model=settings.DEFAULT_LLM_MODEL, 
        temperature=settings.DEFAULT_MODEL_TEMPERATURE,
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    
    context_to_use = state.get("optimized_context", "")
    
    system_prompt = {
        "role": "system", 
        "content": f"Use this context to respond: \n\n{context_to_use}"
    }
    
    response = llm.invoke([system_prompt] + state["messages"])

    print(f"\n\nResponse: {response}\n\n")

    return {"messages": [response]}
