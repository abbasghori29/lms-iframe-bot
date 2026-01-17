"""
Simple debug script to test vector database search
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.config import settings
from train_vector_db import CustomEmbeddings
from langchain_community.vectorstores import FAISS

question = "What is the Target for the Overnight Rate and why is it important?"
k = 5

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
print(f"Loaded {len(vector_store.index_to_docstore_id)} documents")

print(f"\nSearching for: {question}\n")
docs = vector_store.similarity_search_with_score(question, k=k)

print(f"Found {len(docs)} results:\n")
for i, (doc, score) in enumerate(docs, 1):
    print(f"--- Document {i} ---")
    print(f"Score: {score:.4f}")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"Page: {doc.metadata.get('page_number', doc.metadata.get('page', 'N/A'))}")
    print(f"Content (first 500 chars):")
    print(doc.page_content[:500])
    print("\n")
