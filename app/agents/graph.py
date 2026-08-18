from langgraph.graph import StateGraph, START, END
from app.agents.state import AgentState
from app.agents.nodes import retrieval_node, optimization_node, generation_node

# 1. Instantiate the State Graph architecture
builder = StateGraph(AgentState)

# 2. Declare and append all processing nodes
builder.add_node("retrieve_documents", retrieval_node)
builder.add_node("optimize_context", optimization_node)
builder.add_node("generate_response", generation_node)

# 3. Establish the structural execution edges 
builder.add_edge(START, "retrieve_documents")
builder.add_edge("retrieve_documents", "optimize_context")
builder.add_edge("optimize_context", "generate_response")
builder.add_edge("generate_response", END)

# 4. Compile into an executable graph instance
graph = builder.compile()
