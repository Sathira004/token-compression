import os
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from app.core.config import settings

# Bind the system environment tokens that the underlying Pinecone SDK requires
os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY

def run_naive_rag(query: str, k: int = 4) -> str:
    """
    Leverages Pinecone's Integrated Inference API and llama-text-embed-v2 
    to retrieve text data chunks without generating embeddings client-side.
    """
    # Initialize the Pinecone-hosted embedding interface
    pinecone_embeddings = PineconeEmbeddings(
        model=settings.PINECONE_EMBEDDING_MODEL,
        pinecone_api_key=settings.PINECONE_API_KEY
    )
    
    # Connect directly to your existing index using the inference engine instance
    vectorstore = PineconeVectorStore(
        index_name=settings.PINECONE_INDEX_NAME, 
        embedding=pinecone_embeddings
    )
    
    # Perform similarity lookups on raw text query 
    docs = vectorstore.similarity_search(query, k=k)
    
    # Collapse found string chunks cleanly
    return "\n\n".join([doc.page_content for doc in docs])
