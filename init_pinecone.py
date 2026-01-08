"""
Initialize Pinecone index for chat memory using LangChain.
Run this script once to set up your Pinecone index.

Prerequisites:
1. Sign up for free at https://www.pinecone.io/
2. Create an API key from the Pinecone console
3. Add PINECONE_API_KEY to your .env file

Usage:
    python init_pinecone.py
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()


def main():
    print("=" * 60)
    print("Pinecone Memory Index Initialization (LangChain)")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("\n❌ PINECONE_API_KEY not found in environment!")
        print("\nTo set up Pinecone:")
        print("1. Sign up at https://www.pinecone.io/ (free tier available)")
        print("2. Create an API key from the console")
        print("3. Add to your .env file:")
        print("   PINECONE_API_KEY=your-api-key-here")
        print("\nOptional settings (with defaults):")
        print("   PINECONE_INDEX_NAME=csi-chatbot-memory")
        print("   PINECONE_CLOUD=aws")
        print("   PINECONE_REGION=us-east-1")
        sys.exit(1)
    
    print("\n✓ PINECONE_API_KEY found")
    
    # Import after checking API key
    from app.core.config import settings
    
    print(f"\nConfiguration:")
    print(f"  Index name: {settings.PINECONE_INDEX_NAME}")
    print(f"  Cloud: {settings.PINECONE_CLOUD}")
    print(f"  Region: {settings.PINECONE_REGION}")
    
    print("\nInitializing Pinecone with LangChain...")
    
    try:
        from app.services.pinecone_memory import PineconeMemoryService
        
        memory = PineconeMemoryService()
        
        print("\n" + "=" * 60)
        print("Pinecone Memory Index Ready!")
        print("=" * 60)
        
        stats = memory.get_stats()
        print(f"\nIndex Statistics:")
        print(f"  Total conversations: {stats['total_conversations']}")
        print(f"  Total vectors: {stats['index_size']}")
        print(f"  Dimension: {stats['dimension']}")
        if stats.get('index_fullness'):
            print(f"  Index fullness: {stats['index_fullness']:.2%}")
        
        print("\n✓ Pinecone is ready with LangChain integration!")
        print("\nUsing: langchain_pinecone.PineconeVectorStore")
        print("Namespace: conversations")
        
        # Test storing a conversation
        test = input("\nWould you like to store a test conversation? (y/n): ").lower().strip()
        if test == 'y':
            print("Storing test document...")
            conv_id = memory.store_conversation(
                question="What is this test?",
                answer="This is a test conversation to verify Pinecone + LangChain integration is working correctly.",
                user_id="test_user",
                session_id="test_session"
            )
            print(f"✓ Test conversation stored with ID: {conv_id}")
            
            # Test similarity search
            print("\nTesting similarity search...")
            results = memory.search_similar("test conversation", k=1, user_id="test_user")
            if results:
                print(f"✓ Similarity search working! Found: {results[0]['question'][:50]}...")
            
            # Verify retrieval
            print("\nTesting user history retrieval...")
            history = memory.get_user_history("test_user", limit=1)
            if history:
                print(f"✓ User history working! Found {len(history)} conversation(s)")
            
            # Clean up
            cleanup = input("\nClean up test data? (y/n): ").lower().strip()
            if cleanup == 'y':
                memory.clear_user_memory("test_user")
                print("✓ Test data cleaned up")
        
        print("\n" + "=" * 60)
        print("Setup Complete!")
        print("=" * 60)
        print("\nThe chatbot will automatically use Pinecone for memory.")
        print("No local files needed - everything is stored in the cloud!")
        
    except Exception as e:
        import traceback
        print(f"\n❌ Error initializing Pinecone: {e}")
        print("\nFull error:")
        traceback.print_exc()
        print("\nPlease check:")
        print("  1. Your API key is correct")
        print("  2. The region/cloud settings match your Pinecone account")
        print("  3. You have internet connectivity")
        print("  4. Required packages are installed: pip install pinecone langchain-pinecone")
        sys.exit(1)


if __name__ == "__main__":
    main()

