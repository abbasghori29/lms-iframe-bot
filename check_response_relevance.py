"""
Script to check context relevance from vector database.
Simply queries the vector DB and shows top 5 results for manual review.
"""
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.core.config import settings


class ContextChecker:
    """Simple context checker - just queries vector DB"""
    
    def __init__(self):
        """Initialize vector store"""
        self.vector_store = None
        self.embeddings = None
        self._initialize()
    
    def _initialize(self):
        """Initialize embeddings and vector store"""
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=settings.OPENAI_API_KEY,
        )
        
        # Load vector store
        self._load_vector_store()
    
    def _load_vector_store(self):
        """Load the FAISS vector store"""
        vector_store_path = settings.VECTOR_STORE_PATH
        
        if not os.path.exists(vector_store_path):
            raise FileNotFoundError(
                f"Vector store not found at {vector_store_path}. "
                "Please run train_vector_db.py first."
            )
        
        try:
            self.vector_store = FAISS.load_local(
                vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print(f"✓ Loaded vector store with {len(self.vector_store.index_to_docstore_id)} documents")
        except Exception as e:
            raise Exception(f"Error loading vector store: {e}")
    
    def get_context_for_query(self, query: str, k: int = 5) -> List:
        """Retrieve context from vector store for a query"""
        docs = self.vector_store.similarity_search_with_score(query, k=k)
        return docs
    
    def check_context(self, query: str, k: int = 5):
        """Check context for a query - display top k results"""
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"{'='*80}")
        
        # Get context
        print(f"\nRetrieving top {k} results from vector store...")
        docs = self.get_context_for_query(query, k=k)
        
        if not docs:
            print("No results found!")
            return
        
        # Display scores summary
        scores = [score for _, score in docs]
        print(f"\nSimilarity Scores Summary:")
        print(f"  Min: {min(scores):.4f}")
        print(f"  Max: {max(scores):.4f}")
        print(f"  Avg: {sum(scores)/len(scores):.4f}")
        
        # Display each result
        print(f"\n{'='*80}")
        print(f"TOP {len(docs)} RESULTS:")
        print(f"{'='*80}\n")
        
        for i, (doc, score) in enumerate(docs, 1):
            print(f"{'─'*80}")
            print(f"RESULT #{i}")
            print(f"{'─'*80}")
            print(f"Similarity Score: {score:.4f}")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            page_num = doc.metadata.get('page_number') or doc.metadata.get('page', 'N/A')
            print(f"Page: {page_num}")
            print(f"\nContent:")
            print(f"{doc.page_content}")
            print(f"\n")


def main():
    """Main function to check context from vector DB"""
    
    # Test cases from user
    test_queries = [
        "what would happen with the money in my checking account if there were negative rates",
        "why does the central bank charge the commercial banks for storing deposits",
    ]
    
    try:
        checker = ContextChecker()
        
        # Run test cases
        for i, query in enumerate(test_queries, 1):
            print(f"\n\n{'#'*80}")
            print(f"TEST CASE {i}")
            print(f"{'#'*80}")
            
            checker.check_context(query, k=5)
            
            # Wait for user input before next test case
            if i < len(test_queries):
                input("\nPress Enter to continue to next test case...")
        
        # Interactive mode
        print(f"\n\n{'='*80}")
        print("INTERACTIVE MODE")
        print("Enter queries to check context from vector DB")
        print("Type 'exit' to quit")
        print(f"{'='*80}")
        
        while True:
            try:
                print("\n")
                query = input("Enter query: ").strip()
                if not query or query.lower() in ['exit', 'quit', 'q']:
                    break
                
                checker.check_context(query, k=5)
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"Error initializing checker: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

