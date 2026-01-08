"""
Initialize memory vector database for chatbot
Creates an empty FAISS index for storing past Q&A conversations
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.services.memory import MemoryService
from app.core.config import settings


def main():
    """Initialize memory vector database"""
    print("=" * 60)
    print("Initializing Memory Vector Database")
    print("=" * 60)
    
    memory_path = "memory_index"
    
    # Check if already exists
    if os.path.exists(memory_path):
        print(f"\nMemory index already exists at: {memory_path}")
        response = input("Do you want to recreate it? (y/n): ").lower().strip()
        if response == 'y':
            import shutil
            shutil.rmtree(memory_path)
            print("Removed existing memory index")
        else:
            print("Keeping existing memory index")
            return
    
    print("\nCreating memory vector database...")
    print("This will create an empty FAISS index for storing Q&A pairs.")
    
    try:
        # Initialize memory service (this creates the index)
        memory = MemoryService()
        
        # Verify directory was created
        if os.path.exists(memory_path):
            print(f"\n✓ Directory created: {os.path.abspath(memory_path)}")
        else:
            print(f"\n⚠ Warning: Directory not found at {memory_path}")
            print("   It should be created when the first conversation is stored.")
        
        print("\n" + "=" * 60)
        print("Memory Vector Database Initialized Successfully!")
        print("=" * 60)
        print(f"\nMemory index location: {os.path.abspath(memory_path)}/")
        print(f"Embedding model: {settings.EMBEDDING_MODEL}")
        print(f"Embedding API: {settings.EMBEDDING_API_URL}")
        
        # Get stats
        stats = memory.get_stats()
        print(f"\nInitial Stats:")
        print(f"  Total conversations: {stats['total_conversations']}")
        print(f"  Index size: {stats['index_size']}")
        
        # List files in memory directory
        if os.path.exists(memory_path):
            files = os.listdir(memory_path)
            print(f"\nFiles in memory_index/:")
            for file in files:
                file_path = os.path.join(memory_path, file)
                size = os.path.getsize(file_path)
                print(f"  - {file} ({size} bytes)")
        
        print("\n✓ Memory is ready to store conversations!")
        print("\nThe memory will automatically store Q&A pairs when you use the chat API.")
        
    except Exception as e:
        print(f"\n❌ Error initializing memory: {e}")
        print("\nPlease check:")
        print("  1. Embedding API is accessible")
        print("  2. GROQ_API_KEY is set in .env (if needed)")
        print("  3. All dependencies are installed")
        sys.exit(1)


if __name__ == "__main__":
    main()

