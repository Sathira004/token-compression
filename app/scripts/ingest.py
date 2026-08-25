import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from app.core.config import settings

# Bind the system environment tokens required by the Pinecone SDK
os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY

def ingest_root_pdfs():
    # 1. Locate PDF files in the root directory
    pdf_files = glob.glob("*.pdf")
    
    if not pdf_files:
        print("❌ No PDF files found in the root directory.")
        return

    print(f"📂 Found PDF files to process: {pdf_files}")
    all_chunks = []

    # 2. Extract text page-by-page from each PDF
    for pdf_path in pdf_files:
        print(f"📖 Reading content from: {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # 3. Split the text into optimal chunks matching llama-text-embed-v2 recommendations
        # Pinecone recommends 400-500 tokens for maximum retrieval performance
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,       # Approximately 400-500 words/tokens
            chunk_overlap=150      # Safe structural overlap between chunks
        )
        chunks = text_splitter.split_documents(documents)
        
        # Inject metadata to track the original file source easily later
        for chunk in chunks:
            chunk.metadata["source_file"] = pdf_path
            
        all_chunks.extend(chunks)

    print(f"✂️ Successfully parsed documents into {len(all_chunks)} total text blocks.")

    # 4. Initialize the Integrated Inference client wrapper
    print(f"🧠 Connecting to Pinecone Inference Engine with: {settings.PINECONE_EMBEDDING_MODEL}")
    pinecone_embeddings = PineconeEmbeddings(
        model=settings.PINECONE_EMBEDDING_MODEL,
        pinecone_api_key=settings.PINECONE_API_KEY
    )

    # 5. Connect and bulk upsert records natively
    print(f"🚀 Initializing transfer to index: '{settings.PINECONE_INDEX_NAME}'...")
    
    # from_documents handles the upload batches, embedding text automatically cloud-side
    vectorstore = PineconeVectorStore.from_documents(
        documents=all_chunks,
        embedding=pinecone_embeddings,
        index_name=settings.PINECONE_INDEX_NAME
    )

    print("✅ Ingestion complete! Data is indexed and ready for your FastAPI RAG pipelines.")

if __name__ == "__main__":
    ingest_root_pdfs()
