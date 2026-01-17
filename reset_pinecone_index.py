"""
Reset Pinecone index to work with OpenAI embeddings (3072 dimensions).
This will delete the old index and create a new one.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pinecone import Pinecone, ServerlessSpec
from app.core.config import settings


def reset_pinecone_index():
    """Delete old index and info about creating new one"""
    
    if not settings.PINECONE_API_KEY:
        print("ERROR: PINECONE_API_KEY not found!")
        return
    
    print("=" * 60)
    print("Resetting Pinecone Index for OpenAI Embeddings")
    print("=" * 60)
    
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index_name = settings.PINECONE_INDEX_NAME
    
    # Check if index exists
    if pc.has_index(index_name):
        print(f"\nFound existing index: {index_name}")
        
        # Get current stats
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"  Current dimension: {stats.dimension}")
        print(f"  Total vectors: {stats.total_vector_count}")
        
        # Delete the index
        print(f"\nDeleting index: {index_name}...")
        pc.delete_index(index_name)
        print(f"Index deleted!")
    else:
        print(f"\nNo existing index found: {index_name}")
    
    # Create new index with OpenAI dimension (3072)
    print(f"\nCreating new index with dimension 3072...")
    pc.create_index(
        name=index_name,
        dimension=3072,  # OpenAI text-embedding-3-large dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud=settings.PINECONE_CLOUD,
            region=settings.PINECONE_REGION
        )
    )
    print(f"Created new index: {index_name}")
    
    # Verify
    print(f"\nVerifying new index...")
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    print(f"  New dimension: {stats.dimension}")
    print(f"  Total vectors: {stats.total_vector_count}")
    
    print("\n" + "=" * 60)
    print("SUCCESS! Pinecone index recreated with 3072 dimensions.")
    print("=" * 60)


if __name__ == "__main__":
    reset_pinecone_index()
