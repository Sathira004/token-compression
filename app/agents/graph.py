from langgraph.graph import StateGraph, START, END
from app.agents.state import AgentState
from app.agents.nodes import (
    retrieval_node, 
    optimization_node, 
    generation_node, 
    route_query  # <-- Import router function
)

builder = StateGraph(AgentState)

# Nodes
builder.add_node("retrieve_documents", retrieval_node)
builder.add_node("optimize_context", optimization_node)
builder.add_node("generate_response", generation_node)

# Conditional start: choose whether to retrieve or generate directly
builder.add_conditional_edges(
    START,
    route_query,
    {
        "retrieve_documents": "retrieve_documents",
        "generate_response": "optimize_context"  # Route to optimization first
    }
)

# Retrieval pipeline path
builder.add_edge("retrieve_documents", "optimize_context")

# Optimization always precedes generation
builder.add_edge("optimize_context", "generate_response")

# Final end
builder.add_edge("generate_response", END)

graph = builder.compile()
