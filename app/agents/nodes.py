from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from app.agents.state import AgentState
from app.core.config import settings
from app.services.rag_service import run_naive_rag

def route_query(state: AgentState) -> str:
    """Decides whether the user's query requires vector search retrieval."""
    user_message = state["messages"][-1].content if state.get("messages") else ""
    
    # Optional fast heuristic for trivial greetings
    if user_message.lower().strip() in ["hi", "hello", "hey", "who are you?", "thanks", "bye"]:
        return "generate_response"
    
    # Or use the LLM to classify intent
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

    print("\n\nRetrieval Node Triggered!\n\n")
    
    user_message = ""
    if state.get("messages"):
        last_msg = state["messages"][-1]
        user_message = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    raw_context = ""
    if user_message:
        raw_context = run_naive_rag(query=user_message, k=4)

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
