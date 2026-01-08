"""
Train local vector database from PDF documents for chatbot

IMPROVED CHUNKING STRATEGY FOR BETTER RAG:
- Larger chunks (1200 chars) preserve more context for complex questions
- Increased overlap (200 chars) prevents context loss at boundaries
- Smart separators respect document structure (sections, paragraphs, sentences)
- Text cleaning removes noise and formatting issues
- Rich metadata (page numbers, chunk indices) for better retrieval
- Context enrichment adds surrounding chunk info when possible
- Filters out short/noise chunks that hurt retrieval quality

This approach significantly improves answer quality by:
1. Preserving complete thoughts/concepts within chunks
2. Maintaining context across chunk boundaries
3. Providing better metadata for filtering and ranking
4. Reducing noise from headers/footers/short fragments
"""
import os
import requests
import time
import json
from uuid import uuid4
from typing import List, Optional
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS


# Embedding API configuration
API_URL = "https://lamhieu-lightweight-embeddings.hf.space/v1/embeddings"
EMBEDDING_MODEL = "bge-m3"


class CustomEmbeddings(Embeddings):
    """Custom embedding class using the provided API with rate limiting and retry logic"""
    
    def __init__(
        self, 
        api_url: str = API_URL, 
        model: str = EMBEDDING_MODEL,
        delay_between_requests: float = 0.5,  # Delay in seconds between requests
        max_retries: int = 5,
        retry_delay: float = 2.0,  # Initial retry delay in seconds
    ):
        self.api_url = api_url
        self.model = model
        self.delay_between_requests = delay_between_requests
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # Cache to store embedding dimension
        self._dimension = None
        self._last_request_time = 0
    
    def _get_embedding(self, text: str, retry_count: int = 0) -> List[float]:
        """Get embedding for a single text with retry logic and rate limiting"""
        # Rate limiting: ensure minimum delay between requests
        time_since_last = time.time() - self._last_request_time
        if time_since_last < self.delay_between_requests:
            time.sleep(self.delay_between_requests - time_since_last)
        
        payload = {
            "model": self.model,
            "input": text
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            self._last_request_time = time.time()
            
            # Handle rate limiting (429)
            if response.status_code == 429:
                if retry_count < self.max_retries:
                    # Exponential backoff with jitter
                    wait_time = self.retry_delay * (2 ** retry_count) + (time.time() % 1)
                    print(f"  Rate limited (429). Waiting {wait_time:.1f}s before retry {retry_count + 1}/{self.max_retries}...")
                    time.sleep(wait_time)
                    return self._get_embedding(text, retry_count + 1)
                else:
                    raise Exception(f"Max retries reached. Rate limit error: {response.status_code}")
            
            response.raise_for_status()
            data = response.json()
            embedding = data["data"][0]["embedding"]
            
            # Cache dimension on first call
            if self._dimension is None:
                self._dimension = len(embedding)
            
            return embedding
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and retry_count < self.max_retries:
                wait_time = self.retry_delay * (2 ** retry_count) + (time.time() % 1)
                print(f"  HTTP 429 error. Waiting {wait_time:.1f}s before retry {retry_count + 1}/{self.max_retries}...")
                time.sleep(wait_time)
                return self._get_embedding(text, retry_count + 1)
            raise
        except Exception as e:
            if retry_count < self.max_retries:
                wait_time = self.retry_delay * (2 ** retry_count)
                print(f"  Error: {e}. Retrying in {wait_time:.1f}s... (attempt {retry_count + 1}/{self.max_retries})")
                time.sleep(wait_time)
                return self._get_embedding(text, retry_count + 1)
            print(f"Error getting embedding after {self.max_retries} retries: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents with progress tracking"""
        embeddings = []
        total = len(texts)
        start_time = time.time()
        
        for i, text in enumerate(texts, 1):
            # Show progress every 10 documents or on first/last
            if i % 10 == 0 or i == 1 or i == total:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                print(f"Embedding document {i}/{total} (Rate: {rate:.1f}/s, ETA: {eta/60:.1f}m)...")
            
            embeddings.append(self._get_embedding(text))
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._get_embedding(text)
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        if self._dimension is None:
            # Get dimension by embedding a test string
            test_embedding = self.embed_query("test")
            self._dimension = len(test_embedding)
        return self._dimension


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    # Remove special characters that might interfere
    # Keep newlines for structure
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Remove multiple consecutive newlines (keep max 2)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    
    return text.strip()


def load_pdfs(pdf_files: List[str]) -> List[Document]:
    """Load and extract text from PDF files with enhanced metadata"""
    all_documents = []
    
    for pdf_file in pdf_files:
        if not os.path.exists(pdf_file):
            print(f"Warning: PDF file not found: {pdf_file}")
            continue
        
        print(f"Loading PDF: {pdf_file}...")
        try:
            loader = PyPDFLoader(pdf_file)
            documents = loader.load()
            
            # Add enhanced metadata to each document
            for i, doc in enumerate(documents):
                # Clean the text
                doc.page_content = clean_text(doc.page_content)
                
                # Skip empty pages
                if not doc.page_content or len(doc.page_content.strip()) < 10:
                    continue
                
                # Enhanced metadata
                doc.metadata["source"] = os.path.basename(pdf_file)
                doc.metadata["file_path"] = pdf_file
                doc.metadata["file_name"] = os.path.basename(pdf_file)
                doc.metadata["page_number"] = i + 1
                doc.metadata["total_pages"] = len(documents)
                
                # Extract potential chapter/section info from first lines
                first_lines = doc.page_content.split("\n")[:3]
                if first_lines:
                    # Check if first line looks like a heading
                    first_line = first_lines[0].strip()
                    if len(first_line) < 100 and first_line.isupper():
                        doc.metadata["potential_section"] = first_line
            
            # Filter out empty documents
            documents = [doc for doc in documents if doc.page_content and len(doc.page_content.strip()) >= 10]
            
            all_documents.extend(documents)
            print(f"  Loaded {len(documents)} pages from {pdf_file}")
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
            continue
    
    return all_documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks with improved strategy for better RAG retrieval.
    
    Uses larger chunks with smart overlap to preserve context and improve answer quality.
    """
    print("\nSplitting documents into chunks...")
    
    # Improved chunking strategy for better context preservation
    # Larger chunks (1000-1500 chars) work better for technical documents
    # as they preserve more context for complex questions
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,       # Larger chunks for better context (was 800)
        chunk_overlap=200,      # More overlap to prevent context loss (was 150)
        length_function=len,
        separators=[
            "\n\n\n",          # Major section breaks
            "\n\n",            # Paragraph breaks
            "\n",              # Line breaks
            ". ",              # Sentence endings (with space)
            "! ",              # Exclamation endings
            "? ",              # Question endings
            "; ",              # Semicolon breaks
            ", ",              # Comma breaks
            " ",               # Word breaks
            ""                 # Character breaks (last resort)
        ],
        is_separator_regex=False,
        keep_separator=True,   # Keep separators to maintain structure
    )
    
    all_chunks = []
    
    for doc_idx, doc in enumerate(documents):
        # Split this document
        doc_chunks = text_splitter.split_documents([doc])
        
        # Enhance each chunk with metadata
        for chunk_idx, chunk in enumerate(doc_chunks):
            # Skip very short chunks (likely headers/footers/noise)
            if len(chunk.page_content.strip()) < 50:
                continue
            
            # Clean the chunk text
            chunk.page_content = clean_text(chunk.page_content)
            
            # Enhanced metadata for better retrieval
            chunk.metadata.update({
                "chunk_index": chunk_idx,
                "total_chunks_in_doc": len(doc_chunks),
                "chunk_length": len(chunk.page_content),
                "doc_index": doc_idx,
            })
            
            # Add context from surrounding chunks if available
            if chunk_idx > 0:
                prev_chunk = doc_chunks[chunk_idx - 1]
                # Store last sentence of previous chunk for context
                prev_sentences = prev_chunk.page_content.split(". ")
                if prev_sentences:
                    chunk.metadata["prev_context"] = prev_sentences[-1][:100]
            
            all_chunks.append(chunk)
    
    # Filter out chunks that are too short or likely noise
    filtered_chunks = []
    for chunk in all_chunks:
        content = chunk.page_content.strip()
        
        # Skip if too short
        if len(content) < 50:
            continue
        
        # Skip if it's mostly numbers or special chars (likely headers/footers)
        alpha_chars = sum(1 for c in content if c.isalpha())
        if len(content) > 0 and alpha_chars / len(content) < 0.3:
            continue
        
        filtered_chunks.append(chunk)
    
    print(f"Created {len(filtered_chunks)} chunks from {len(documents)} documents")
    print(f"Average chunk size: {sum(len(c.page_content) for c in filtered_chunks) // len(filtered_chunks) if filtered_chunks else 0} characters")
    
    return filtered_chunks


def enrich_chunks_with_context(chunks: List[Document]) -> List[Document]:
    """
    Enrich chunks with surrounding context for better retrieval.
    This helps maintain context when chunks are retrieved individually.
    """
    print("\nEnriching chunks with context...")
    
    enriched_chunks = []
    
    for i, chunk in enumerate(chunks):
        # Create a context-aware version
        context_parts = []
        
        # Add previous chunk context if available (last 100 chars)
        if i > 0:
            prev_chunk = chunks[i - 1]
            if prev_chunk.metadata.get("source") == chunk.metadata.get("source"):
                prev_text = prev_chunk.page_content[-100:].strip()
                if prev_text:
                    context_parts.append(f"[Previous context: {prev_text}]")
        
        # Add current chunk
        context_parts.append(chunk.page_content)
        
        # Add next chunk context if available (first 100 chars)
        if i < len(chunks) - 1:
            next_chunk = chunks[i + 1]
            if next_chunk.metadata.get("source") == chunk.metadata.get("source"):
                next_text = next_chunk.page_content[:100].strip()
                if next_text:
                    context_parts.append(f"[Following context: {next_text}]")
        
        # Create enriched document
        enriched_content = " ".join(context_parts)
        
        # Only add context if it doesn't make chunk too large
        if len(enriched_content) <= 2000:  # Reasonable limit
            chunk.page_content = enriched_content
            chunk.metadata["has_context"] = True
        else:
            chunk.metadata["has_context"] = False
        
        enriched_chunks.append(chunk)
    
    print(f"Enriched {sum(1 for c in enriched_chunks if c.metadata.get('has_context'))} chunks with context")
    return enriched_chunks


def create_vector_store(documents: List[Document], embeddings: CustomEmbeddings) -> FAISS:
    """Create FAISS vector store from documents"""
    print("\nCreating vector store...")
    
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
    
    # Generate UUIDs for documents
    total_docs = len(documents)
    print(f"\nAdding {total_docs} documents to vector store...")
    print("This will take a while due to rate limiting (~0.6s per document).")
    print("Estimated time: ~{:.1f} minutes".format(total_docs * 0.6 / 60))
    print("The rate limiter will automatically handle 429 errors with retries.")
    
    uuids = [str(uuid4()) for _ in range(total_docs)]
    
    try:
        # Add documents - rate limiting is handled in CustomEmbeddings
        vector_store.add_documents(documents=documents, ids=uuids)
        print("Vector store created successfully!")
        return vector_store
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        print("Saving partial vector store...")
        vector_store.save_local("faiss_index")
        print("Partial vector store saved to faiss_index/")
        print("Note: You'll need to restart from the beginning.")
        raise
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Attempting to save partial vector store...")
        try:
            vector_store.save_local("faiss_index")
            print("Partial vector store saved to faiss_index/")
        except:
            print("Could not save partial vector store.")
        raise


def save_vector_store(vector_store: FAISS, save_path: str = "faiss_index"):
    """Save vector store to disk"""
    print(f"\nSaving vector store to {save_path}...")
    vector_store.save_local(save_path)
    print(f"Vector store saved to {save_path}")


def test_vector_store(vector_store: FAISS, embeddings: CustomEmbeddings, query: str = "What is CSI?"):
    """Test the vector store with a sample query"""
    print(f"\nTesting vector store with query: '{query}'...")
    
    # Test with similarity search
    results = vector_store.similarity_search(query, k=5)
    
    print(f"\nFound {len(results)} results:")
    for i, res in enumerate(results, 1):
        content_preview = res.page_content[:300].replace("\n", " ")
        print(f"\n{'='*60}")
        print(f"Result {i}:")
        print(f"Content: {content_preview}...")
        print(f"Source: {res.metadata.get('source', 'Unknown')}")
        print(f"Page: {res.metadata.get('page_number', res.metadata.get('page', 'N/A'))}")
        print(f"Chunk: {res.metadata.get('chunk_index', 'N/A')}/{res.metadata.get('total_chunks_in_doc', 'N/A')}")
        print(f"Length: {res.metadata.get('chunk_length', len(res.page_content))} chars")
    
    # Also test with similarity search with scores
    print(f"\n{'='*60}")
    print("Testing with similarity scores:")
    results_with_scores = vector_store.similarity_search_with_score(query, k=3)
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   {doc.page_content[:150]}...")


def main():
    """Main function to train vector database"""
    print("=" * 60)
    print("Training Vector Database from PDF Documents")
    print("=" * 60)
    
    # PDF files to process
    pdf_files = [
        "FIC-CSI-EN-2025.pdf",
        "PDF complet de CSI.pdf"
    ]
    
    # Initialize embeddings with rate limiting
    print("\nInitializing embeddings...")
    # Increase delay to avoid rate limits (0.5s = 2 requests/second max)
    # Adjust based on API limits
    embeddings = CustomEmbeddings(
        delay_between_requests=0.6,  # Slightly slower to be safe
        max_retries=10,  # More retries for rate limits
        retry_delay=5.0,  # Start with 5s delay on rate limit
    )
    
    # Load PDFs
    print("\n" + "=" * 60)
    print("Step 1: Loading PDF files")
    print("=" * 60)
    documents = load_pdfs(pdf_files)
    
    if not documents:
        print("No documents loaded. Exiting.")
        return
    
    print(f"\nTotal pages loaded: {len(documents)}")
    
    # Split documents into chunks
    print("\n" + "=" * 60)
    print("Step 2: Splitting documents into chunks")
    print("=" * 60)
    chunks = split_documents(documents)
    
    # Enrich chunks with context (optional but recommended)
    print("\n" + "=" * 60)
    print("Step 2.5: Enriching chunks with context")
    print("=" * 60)
    chunks = enrich_chunks_with_context(chunks)
    
    # Create vector store
    print("\n" + "=" * 60)
    print("Step 3: Creating vector store")
    print("=" * 60)
    vector_store = create_vector_store(chunks, embeddings)
    
    # Save vector store
    print("\n" + "=" * 60)
    print("Step 4: Saving vector store")
    print("=" * 60)
    save_vector_store(vector_store)
    
    # Test vector store
    print("\n" + "=" * 60)
    print("Step 5: Testing vector store")
    print("=" * 60)
    test_vector_store(vector_store, embeddings)
    
    print("\n" + "=" * 60)
    print("Vector database training completed successfully!")
    print("=" * 60)
    print(f"\nVector store saved to: faiss_index/")
    print("You can now load it in your FastAPI app using:")
    print("  vector_store = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)")


if __name__ == "__main__":
    main()

