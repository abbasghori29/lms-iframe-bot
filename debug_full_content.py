"""
Debug to show FULL content of search results
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.config import settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

question = "What is the Target for the Overnight Rate and why is it important?"
k = 3

print("Loading OpenAI embeddings...")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=settings.OPENAI_API_KEY,
)

print(f"Loading vector store from: faiss_index_openai")
vector_store = FAISS.load_local(
    "faiss_index_openai",
    embeddings,
    allow_dangerous_deserialization=True,
)

print(f"\nQuestion: {question}\n")
print("=" * 80)

results = vector_store.similarity_search_with_score(question, k=k)

for i, (doc, score) in enumerate(results, 1):
    print(f"\n{'='*80}")
    print(f"RESULT {i} - Score: {score:.4f}")
    print(f"Source: {doc.metadata.get('source')}, Page: {doc.metadata.get('page_number')}")
    print(f"{'='*80}")
    print("\nFULL CONTENT:")
    print("-" * 40)
    print(doc.page_content)
    print("-" * 40)
