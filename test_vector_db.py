"""
Test the trained vector database with various queries
"""
import os
from train_vector_db import CustomEmbeddings
from langchain_community.vectorstores import FAISS


def load_vector_store(embeddings: CustomEmbeddings, path: str = "faiss_index"):
    """Load the vector store from disk"""
    if not os.path.exists(path):
        print(f"Error: Vector store not found at {path}")
        print("Please run train_vector_db.py first to create the vector store.")
        return None
    
    try:
        print(f"Loading vector store from {path}...")
        vector_store = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✓ Loaded vector store with {len(vector_store.index_to_docstore_id)} documents")
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None


def test_query(vector_store: FAISS, query: str, k: int = 10):
    """Test a query and display top k results"""
    print("\n" + "=" * 80)
    print(f"Query: '{query}'")
    print("=" * 80)
    
    # Get results with scores
    results_with_scores = vector_store.similarity_search_with_score(query, k=k)
    
    print(f"\nTop {len(results_with_scores)} Results (lower score = more similar):\n")
    
    for i, (doc, score) in enumerate(results_with_scores, 1):
        # Clean up the content for display
        content = doc.page_content.replace("\n", " ").strip()
        
        # Truncate if too long
        if len(content) > 400:
            content = content[:400] + "..."
        
        print(f"{'─' * 80}")
        print(f"Result {i} (Score: {score:.2f})")
        print(f"Content: {content}")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Page: {doc.metadata.get('page_number', doc.metadata.get('page', 'N/A'))}")
        print(f"Chunk: {doc.metadata.get('chunk_index', 'N/A')}/{doc.metadata.get('total_chunks_in_doc', 'N/A')}")
        print(f"Length: {doc.metadata.get('chunk_length', len(doc.page_content))} chars")
    
    print(f"\n{'─' * 80}\n")


def analyze_results(results_with_scores):
    """Analyze the quality of results"""
    if not results_with_scores:
        return
    
    scores = [score for _, score in results_with_scores]
    avg_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)
    
    print(f"Score Analysis:")
    print(f"  Best (most similar): {min_score:.2f}")
    print(f"  Worst (least similar): {max_score:.2f}")
    print(f"  Average: {avg_score:.2f}")
    print(f"  Score range: {max_score - min_score:.2f}")
    print()


def main():
    """Main test function"""
    print("=" * 80)
    print("Vector Database Test Suite")
    print("=" * 80)
    
    # Initialize embeddings
    print("\nInitializing embeddings...")
    embeddings = CustomEmbeddings(
        delay_between_requests=0.1,  # Faster for testing (fewer requests)
    )
    
    # Load vector store
    vector_store = load_vector_store(embeddings)
    if not vector_store:
        return
    
    # Test queries
    test_queries = [
        "What is CSI?",
        "What courses does CSI offer?",
        "CSI certification requirements",
        "How to become certified by CSI?",
        "CSI training programs",
        "What are the benefits of CSI certification?",
    ]
    
    print("\n" + "=" * 80)
    print("Running Test Queries (Top 10 Results Each)")
    print("=" * 80)
    
    for query in test_queries:
        test_query(vector_store, query, k=10)
        # Analyze results
        results_with_scores = vector_store.similarity_search_with_score(query, k=10)
        analyze_results(results_with_scores)
    
    # Interactive mode
    print("\n" + "=" * 80)
    print("Interactive Mode - Enter your own queries")
    print("Type 'exit' or 'quit' to stop")
    print("=" * 80)
    
    while True:
        try:
            query = input("\nEnter query: ").strip()
            if not query:
                continue
            if query.lower() in ['exit', 'quit', 'q']:
                break
            
            test_query(vector_store, query, k=10)
            results_with_scores = vector_store.similarity_search_with_score(query, k=10)
            analyze_results(results_with_scores)
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()

