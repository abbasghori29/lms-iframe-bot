"""
Train vector database using OpenAI embeddings for better semantic search quality.
This creates a separate FAISS index with OpenAI's text-embedding-3-large model.
"""
import os
import sys
from pathlib import Path
from typing import List
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.config import settings


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    text = " ".join(text.split())
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def load_pdfs(pdf_files: List[str]) -> List[Document]:
    """Load and extract text from PDF files"""
    all_documents = []
    
    for pdf_file in pdf_files:
        if not os.path.exists(pdf_file):
            print(f"Warning: PDF file not found: {pdf_file}")
            continue
        
        print(f"Loading PDF: {pdf_file}...")
        try:
            loader = PyPDFLoader(pdf_file)
            documents = loader.load()
            
            for i, doc in enumerate(documents):
                doc.page_content = clean_text(doc.page_content)
                
                if not doc.page_content or len(doc.page_content.strip()) < 10:
                    continue
                
                doc.metadata["source"] = os.path.basename(pdf_file)
                doc.metadata["file_path"] = pdf_file
                doc.metadata["page_number"] = i + 1
                doc.metadata["total_pages"] = len(documents)
            
            documents = [doc for doc in documents if doc.page_content and len(doc.page_content.strip()) >= 10]
            all_documents.extend(documents)
            print(f"  Loaded {len(documents)} pages from {pdf_file}")
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
            continue
    
    return all_documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks"""
    print("\nSplitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        keep_separator=True,
    )
    
    all_chunks = []
    
    for doc_idx, doc in enumerate(documents):
        doc_chunks = text_splitter.split_documents([doc])
        
        for chunk_idx, chunk in enumerate(doc_chunks):
            if len(chunk.page_content.strip()) < 50:
                continue
            
            chunk.page_content = clean_text(chunk.page_content)
            chunk.metadata.update({
                "chunk_index": chunk_idx,
                "total_chunks_in_doc": len(doc_chunks),
                "chunk_length": len(chunk.page_content),
                "doc_index": doc_idx,
            })
            all_chunks.append(chunk)
    
    # Filter noise
    filtered_chunks = []
    for chunk in all_chunks:
        content = chunk.page_content.strip()
        if len(content) < 50:
            continue
        alpha_chars = sum(1 for c in content if c.isalpha())
        if len(content) > 0 and alpha_chars / len(content) < 0.3:
            continue
        filtered_chunks.append(chunk)
    
    print(f"Created {len(filtered_chunks)} chunks from {len(documents)} documents")
    return filtered_chunks


def enrich_chunks_with_context(chunks: List[Document]) -> List[Document]:
    """Add surrounding context to chunks"""
    print("\nEnriching chunks with context...")
    
    for i, chunk in enumerate(chunks):
        context_parts = []
        
        if i > 0:
            prev_chunk = chunks[i - 1]
            if prev_chunk.metadata.get("source") == chunk.metadata.get("source"):
                prev_text = prev_chunk.page_content[-100:].strip()
                if prev_text:
                    context_parts.append(f"[Previous context: {prev_text}]")
        
        context_parts.append(chunk.page_content)
        
        if i < len(chunks) - 1:
            next_chunk = chunks[i + 1]
            if next_chunk.metadata.get("source") == chunk.metadata.get("source"):
                next_text = next_chunk.page_content[:100].strip()
                if next_text:
                    context_parts.append(f"[Following context: {next_text}]")
        
        enriched_content = " ".join(context_parts)
        if len(enriched_content) <= 2000:
            chunk.page_content = enriched_content
            chunk.metadata["has_context"] = True
        else:
            chunk.metadata["has_context"] = False
    
    print(f"Enriched {sum(1 for c in chunks if c.metadata.get('has_context'))} chunks with context")
    return chunks


def create_vector_store_openai(documents: List[Document], embeddings: OpenAIEmbeddings) -> FAISS:
    """Create FAISS vector store using OpenAI embeddings"""
    print("\nCreating vector store with OpenAI embeddings...")
    
    # Get embedding dimension
    print("Getting embedding dimension...")
    test_embedding = embeddings.embed_query("test")
    dimension = len(test_embedding)
    print(f"Embedding dimension: {dimension}")
    
    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)
    
    # Create vector store
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    
    total_docs = len(documents)
    print(f"\nAdding {total_docs} documents to vector store...")
    print("Using OpenAI embeddings - this is faster than the HuggingFace API!")
    
    # Process in batches for efficiency
    batch_size = 100
    uuids = [str(uuid4()) for _ in range(total_docs)]
    
    for i in range(0, total_docs, batch_size):
        batch_end = min(i + batch_size, total_docs)
        batch_docs = documents[i:batch_end]
        batch_ids = uuids[i:batch_end]
        
        print(f"  Processing batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} ({i+1}-{batch_end}/{total_docs})...")
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)
    
    print("Vector store created successfully!")
    return vector_store


def test_vector_store(vector_store: FAISS, test_query: str = "What is a mutual fund?"):
    """Test the vector store with a sample query"""
    print(f"\nTesting vector store with query: '{test_query}'...")
    
    results_with_scores = vector_store.similarity_search_with_score(test_query, k=5)
    
    print(f"\nFound {len(results_with_scores)} results:")
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"\n{'='*60}")
        print(f"Result {i}: Score = {score:.4f} (lower is better)")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page_number', 'N/A')}")
        print(f"Content: {doc.page_content[:300]}...")
    
    # Check if scores are good
    if results_with_scores:
        best_score = results_with_scores[0][1]
        if best_score < 0.5:
            print(f"\n*** EXCELLENT! Best score {best_score:.4f} indicates high-quality semantic matching! ***")
        elif best_score < 1.0:
            print(f"\n*** GOOD! Best score {best_score:.4f} indicates decent semantic matching. ***")
        else:
            print(f"\n*** Score {best_score:.4f} - OpenAI embeddings should give better results. ***")


def main():
    """Main function to train vector database with OpenAI embeddings"""
    print("=" * 70)
    print("Training Vector Database with OpenAI Embeddings")
    print("=" * 70)
    
    # Check for OpenAI API key
    if not settings.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found in environment!")
        print("Please set it in your .env file.")
        return
    
    print(f"\nUsing OpenAI API Key: {settings.OPENAI_API_KEY[:8]}...")
    
    # Initialize OpenAI embeddings
    print("\nInitializing OpenAI embeddings (text-embedding-3-large)...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=settings.OPENAI_API_KEY,
    )
    
    # PDF files to process
    pdf_files = [
        "FIC-CSI-EN-2025.pdf",
        "PDF complet de CSI.pdf"
    ]
    
    # Load PDFs
    print("\n" + "=" * 60)
    print("Step 1: Loading PDF files")
    print("=" * 60)
    documents = load_pdfs(pdf_files)
    
    if not documents:
        print("No documents loaded. Exiting.")
        return
    
    print(f"\nTotal pages loaded: {len(documents)}")
    
    # Split documents
    print("\n" + "=" * 60)
    print("Step 2: Splitting documents into chunks")
    print("=" * 60)
    chunks = split_documents(documents)
    
    # Enrich with context
    print("\n" + "=" * 60)
    print("Step 3: Enriching chunks with context")
    print("=" * 60)
    chunks = enrich_chunks_with_context(chunks)
    
    # Create vector store
    print("\n" + "=" * 60)
    print("Step 4: Creating vector store with OpenAI embeddings")
    print("=" * 60)
    vector_store = create_vector_store_openai(chunks, embeddings)
    
    # Save vector store
    save_path = "faiss_index_openai"
    print("\n" + "=" * 60)
    print(f"Step 5: Saving vector store to {save_path}")
    print("=" * 60)
    vector_store.save_local(save_path)
    print(f"Vector store saved to {save_path}/")
    
    # Test vector store
    print("\n" + "=" * 60)
    print("Step 6: Testing vector store")
    print("=" * 60)
    test_vector_store(vector_store, "What is a mutual fund, and how does it provide value to a small investor?")
    
    print("\n" + "=" * 60)
    print("SUCCESS! OpenAI embeddings vector store created!")
    print("=" * 60)
    print(f"\nTo use this new vector store, update VECTOR_STORE_PATH in .env to: {save_path}")
    print("Or update the chatbot.py to use OpenAI embeddings and this new index.")


if __name__ == "__main__":
    main()
