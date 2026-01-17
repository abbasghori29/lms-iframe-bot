"""
Debug script to search for specific content in the vector store
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.config import settings
from train_vector_db import CustomEmbeddings
from langchain_community.vectorstores import FAISS


def search_by_content(search_text: str, max_results: int = 20):
    """Search for specific content in all indexed documents"""
    print("Loading embeddings...")
    embeddings = CustomEmbeddings(
        api_url=settings.EMBEDDING_API_URL,
        model=settings.EMBEDDING_MODEL,
        delay_between_requests=0.1,
    )

    print(f"Loading vector store from: {settings.VECTOR_STORE_PATH}")
    vector_store = FAISS.load_local(
        settings.VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    
    # Access the docstore directly to search all documents
    docstore = vector_store.docstore
    index_to_id = vector_store.index_to_docstore_id
    
    print(f"\nSearching {len(index_to_id)} documents for: '{search_text[:50]}...'\n")
    
    found_docs = []
    for idx, doc_id in index_to_id.items():
        doc = docstore.search(doc_id)
        if doc and search_text.lower() in doc.page_content.lower():
            found_docs.append({
                'idx': idx,
                'doc_id': doc_id,
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page_number', doc.metadata.get('page', 'N/A')),
                'content': doc.page_content[:500]
            })
            
            if len(found_docs) >= max_results:
                break
    
    if found_docs:
        print(f"FOUND {len(found_docs)} documents containing the text:\n")
        for d in found_docs:
            print(f"--- Index: {d['idx']} ---")
            print(f"Source: {d['source']}, Page: {d['page']}")
            print(f"Content preview:\n{d['content']}\n")
    else:
        print("NOT FOUND in any indexed document!")
        print("\nThis means either:")
        print("1. The page wasn't indexed properly")
        print("2. The text was split across chunks")
        print("3. The PDF extraction failed for that page")
    
    return found_docs


def check_page_exists(source_file: str, page_number: int):
    """Check if a specific page was indexed"""
    print(f"\nChecking if {source_file} page {page_number} is indexed...")
    
    embeddings = CustomEmbeddings(
        api_url=settings.EMBEDDING_API_URL,
        model=settings.EMBEDDING_MODEL,
        delay_between_requests=0.1,
    )

    vector_store = FAISS.load_local(
        settings.VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    
    docstore = vector_store.docstore
    index_to_id = vector_store.index_to_docstore_id
    
    found = []
    for idx, doc_id in index_to_id.items():
        doc = docstore.search(doc_id)
        if doc:
            doc_source = doc.metadata.get('source', '')
            doc_page = doc.metadata.get('page_number', doc.metadata.get('page', -1))
            if source_file in doc_source and doc_page == page_number:
                found.append({
                    'idx': idx,
                    'content': doc.page_content[:300]
                })
    
    if found:
        print(f"FOUND {len(found)} chunks from {source_file} page {page_number}:")
        for f in found[:5]:  # Show first 5
            print(f"\n  Chunk at index {f['idx']}:")
            print(f"  {f['content'][:200]}...")
    else:
        print(f"Page {page_number} from {source_file} NOT FOUND in index!")
    
    return found


if __name__ == "__main__":
    # Search for the exact phrase from the PDF
    print("=" * 80)
    print("SEARCHING FOR MUTUAL FUND DEFINITION")
    print("=" * 80)
    
    # Search for key phrases that should be in the content
    search_phrases = [
        "mutual fund is an investment vehicle that pools contributions",
        "pools contributions from investors",
        "variety of securities, including stocks, bonds",
        "Professional money managers manage the fund",
    ]
    
    for phrase in search_phrases:
        print(f"\n{'='*60}")
        print(f"Searching for: '{phrase}'")
        print("=" * 60)
        found = search_by_content(phrase, max_results=3)
        if not found:
            print(">>> NOT INDEXED!")
    
    # Also check early pages of the English PDF
    print("\n\n" + "=" * 80)
    print("CHECKING EARLY PAGES OF FIC-CSI-EN-2025.pdf")
    print("=" * 80)
    
    for page in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        check_page_exists("FIC-CSI-EN-2025.pdf", page)
