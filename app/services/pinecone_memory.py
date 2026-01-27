"""
Pinecone-based Memory Service using LangChain's PineconeVectorStore.
Stores chat conversations with per-user metadata filtering.
"""
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings


class PineconeMemoryService:
    """
    Memory service using LangChain's PineconeVectorStore.
    Supports per-user memory isolation via metadata filtering.
    """
    
    def __init__(self, embeddings=None):
        # Use OpenAI embeddings for compatibility with 3072-dimension index
        self.embeddings = embeddings or OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=settings.OPENAI_API_KEY,
        )
        self.pc: Optional[Pinecone] = None
        self.index = None
        self.vector_store: Optional[PineconeVectorStore] = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Pinecone connection and vector store"""
        if not settings.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        # Get embedding dimension
        test_embedding = self.embeddings.embed_query("test")
        dimension = len(test_embedding)
        
        # Check if index exists, create if not
        if not self.pc.has_index(settings.PINECONE_INDEX_NAME):
            print(f"Creating Pinecone index: {settings.PINECONE_INDEX_NAME}")
            self.pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.PINECONE_CLOUD,
                    region=settings.PINECONE_REGION
                )
            )
            print(f"✓ Created Pinecone index: {settings.PINECONE_INDEX_NAME}")
        
        # Connect to index
        self.index = self.pc.Index(settings.PINECONE_INDEX_NAME)
        
        # Create LangChain vector store
        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            namespace="conversations"
        )
        
        # Get index stats
        stats = self.index.describe_index_stats()
        print(f"✓ Connected to Pinecone index: {settings.PINECONE_INDEX_NAME}")
        print(f"  Total vectors: {stats.total_vector_count}")
    
    def store_conversation(
        self,
        question: str,
        answer: str,
        sources: List[Dict] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Store a Q&A pair in Pinecone using LangChain Documents.
        
        Args:
            question: User's question
            answer: AI's answer
            sources: Optional list of sources used
            user_id: Anonymous user ID
            session_id: Session ID
        
        Returns:
            Conversation ID
        """
        conv_id = str(uuid4())
        timestamp = datetime.now().isoformat()
        
        # Create combined content for embedding
        content = f"Question: {question}\n\nAnswer: {answer[:500]}"
        
        # Create LangChain Document with metadata
        document = Document(
            page_content=content,
            metadata={
                "conv_id": conv_id,
                "user_id": user_id or "anonymous",
                "session_id": session_id or "",
                "question": question[:1000],  # Pinecone metadata limit
                "answer": answer[:2000],
                "timestamp": timestamp,
                "type": "qa_pair",
                "sources_count": len(sources) if sources else 0,
            }
        )
        
        # Add to vector store using LangChain
        self.vector_store.add_documents(documents=[document], ids=[conv_id])
        
        return conv_id
    
    def search_similar(
        self,
        query: str,
        k: int = 3,
        user_id: Optional[str] = None,
        include_global: bool = False
    ) -> List[Dict]:
        """
        Search for similar past conversations using LangChain.
        
        Args:
            query: Search query
            k: Number of results
            user_id: Filter by user ID
            include_global: Whether to include other users' conversations
        
        Returns:
            List of similar conversations
        """
        # Build filter
        filter_dict = None
        if user_id and not include_global:
            filter_dict = {"user_id": {"$eq": user_id}}
        
        # Use LangChain's similarity search with filter
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict
        )
        
        # Format results
        conversations = []
        for doc, score in results:
            conversations.append({
                "question": doc.metadata.get("question", ""),
                "answer": doc.metadata.get("answer", ""),
                "score": float(score),
                "timestamp": doc.metadata.get("timestamp", ""),
                "user_id": doc.metadata.get("user_id", "anonymous"),
                "session_id": doc.metadata.get("session_id", ""),
            })
        
        return conversations
    
    def get_memory_context(
        self,
        query: str,
        k: int = 2,
        user_id: Optional[str] = None
    ) -> str:
        """
        Get formatted memory context for a query.
        
        Args:
            query: Current query
            k: Number of similar conversations
            user_id: User ID to filter by
        
        Returns:
            Formatted context string
        """
        # First try user's own history
        similar = self.search_similar(query, k=k, user_id=user_id, include_global=False)
        
        # If not enough results, include global
        if len(similar) < k and user_id:
            global_similar = self.search_similar(query, k=k-len(similar), include_global=True)
            similar.extend(global_similar)
        
        if not similar:
            return ""
        
        context_parts = ["Previous relevant conversations:"]
        for i, conv in enumerate(similar, 1):
            # Cosine similarity: higher is better (0.7+ is good)
            if conv["score"] > 0.5:
                is_own = " (from your history)" if user_id and conv.get("user_id") == user_id else ""
                context_parts.append(
                    f"\n[Past Q&A {i}{is_own}]\n"
                    f"User asked: {conv['question']}\n"
                    f"Answer: {conv['answer'][:300]}..."
                )
        
        if len(context_parts) == 1:
            return ""
        
        return "\n".join(context_parts)
    
    def get_user_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        """
        Get a user's conversation history using similarity search with filter.
        """
        # Use similarity search with user filter
        results = self.vector_store.similarity_search(
            query="conversation history",  # Generic query
            k=limit,
            filter={"user_id": {"$eq": user_id}}
        )
        
        conversations = []
        for doc in results:
            conversations.append({
                "id": doc.metadata.get("conv_id", ""),
                "question": doc.metadata.get("question", ""),
                "answer": doc.metadata.get("answer", ""),
                "timestamp": doc.metadata.get("timestamp", ""),
                "session_id": doc.metadata.get("session_id", ""),
            })
        
        # Sort by timestamp descending
        conversations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return conversations[:limit]
    
    def get_session_history(self, user_id: str, session_id: str) -> List[Dict]:
        """
        Get conversation history for a specific session.
        """
        results = self.vector_store.similarity_search(
            query="session conversation",
            k=100,
            filter={
                "$and": [
                    {"user_id": {"$eq": user_id}},
                    {"session_id": {"$eq": session_id}}
                ]
            }
        )
        
        conversations = []
        for doc in results:
            conversations.append({
                "id": doc.metadata.get("conv_id", ""),
                "question": doc.metadata.get("question", ""),
                "answer": doc.metadata.get("answer", ""),
                "timestamp": doc.metadata.get("timestamp", ""),
            })
        
        # Sort by timestamp ascending (chronological)
        conversations.sort(key=lambda x: x.get("timestamp", ""))
        
        return conversations
    
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        stats = self.index.describe_index_stats()
        
        # Get namespace-specific stats
        ns_stats = stats.namespaces.get("conversations", {})
        
        return {
            "total_conversations": ns_stats.get("vector_count", 0) if ns_stats else stats.total_vector_count,
            "index_size": stats.total_vector_count,
            "index_fullness": getattr(stats, 'index_fullness', None),
            "dimension": stats.dimension,
        }
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get statistics for a specific user"""
        history = self.get_user_history(user_id, limit=100)
        
        if not history:
            return {
                "user_id": user_id,
                "exists": False,
                "conversation_count": 0,
                "first_seen": None,
                "last_seen": None,
                "session_count": 0,
            }
        
        timestamps = [c.get("timestamp") for c in history if c.get("timestamp")]
        sessions = set(c.get("session_id") for c in history if c.get("session_id"))
        
        return {
            "user_id": user_id,
            "exists": True,
            "conversation_count": len(history),
            "first_seen": min(timestamps) if timestamps else None,
            "last_seen": max(timestamps) if timestamps else None,
            "session_count": len(sessions),
        }
    
    def clear_user_memory(self, user_id: str) -> bool:
        """Clear all memory for a specific user"""
        try:
            history = self.get_user_history(user_id, limit=1000)
            ids_to_delete = [c["id"] for c in history if c.get("id")]
            
            if ids_to_delete:
                # Delete using Pinecone index directly (LangChain doesn't expose delete)
                for i in range(0, len(ids_to_delete), 100):
                    batch = ids_to_delete[i:i+100]
                    self.index.delete(ids=batch, namespace="conversations")
            
            return True
        except Exception as e:
            print(f"Error clearing user memory: {e}")
            return False
    
    def clear_all_memory(self) -> bool:
        """Clear all conversations"""
        try:
            self.index.delete(delete_all=True, namespace="conversations")
            return True
        except Exception as e:
            print(f"Error clearing all memory: {e}")
            return False


# Singleton instance
_pinecone_memory_service: Optional[PineconeMemoryService] = None


def get_pinecone_memory_service(embeddings=None) -> PineconeMemoryService:
    """Get or create Pinecone memory service instance"""
    global _pinecone_memory_service
    if _pinecone_memory_service is None:
        _pinecone_memory_service = PineconeMemoryService(embeddings=embeddings)
    return _pinecone_memory_service
