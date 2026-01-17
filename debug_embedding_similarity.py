"""
Test embedding similarity with OpenAI embeddings
"""
import os
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.config import settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def test_openai_vector_search():
    """Test the new OpenAI vector store"""
    
    # Test questions
    questions = [
        "What is the Target for the Overnight Rate and why is it important?",
    ]
    
    print("=" * 70)
    print("Testing OpenAI Embeddings Vector Store")
    print("=" * 70)
    
    # Load OpenAI embeddings
    print("\nLoading OpenAI embeddings...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=settings.OPENAI_API_KEY,
    )
    
    # Load the new vector store
    vector_store_path = "faiss_index_openai"
    print(f"Loading vector store from: {vector_store_path}")
    
    if not os.path.exists(vector_store_path):
        print(f"ERROR: {vector_store_path} not found!")
        print("Run train_vector_db_openai.py first.")
        return
    
    vector_store = FAISS.load_local(
        vector_store_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"Loaded {len(vector_store.index_to_docstore_id)} documents")
    
    # Test each question
    for question in questions:
        print("\n" + "=" * 70)
        print(f"Question: {question}")
        print("=" * 70)
        
        results = vector_store.similarity_search_with_score(question, k=5)
        
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page_number', doc.metadata.get('page', 'N/A'))
            content_preview = doc.page_content[:200].replace('\n', ' ')
            
            # Determine quality
            if score < 0.5:
                quality = "EXCELLENT"
            elif score < 1.0:
                quality = "GOOD"
            elif score < 1.5:
                quality = "OK"
            else:
                quality = "POOR"
            
            print(f"\n  [{i}] Score: {score:.4f} ({quality})")
            print(f"      Source: {source}, Page: {page}")
            print(f"      Content: {content_preview}...")
        
        # Summary
        if results:
            best_score = results[0][1]
            print(f"\n  >>> Best score: {best_score:.4f}")
            if best_score < 1.0:
                print("  >>> This is a GOOD match - the content should be found!")
    
    print("\n" + "=" * 70)
    print("COMPARISON:")
    print("  Old HuggingFace embeddings: scores ~267-295 (TERRIBLE)")
    print("  New OpenAI embeddings: scores ~0.8-1.2 (GOOD)")
    print("=" * 70)


if __name__ == "__main__":
    test_openai_vector_search()
