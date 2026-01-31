"""
Memory service using FAISS for storing and retrieving past Q&A pairs.
Supports per-user memory isolation for anonymous users.
"""
import os
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from app.core.config import settings

# Import from project root
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from train_vector_db import CustomEmbeddings


class MemoryService:
    """
    Memory service that stores Q&A pairs in a FAISS index.
    Enables retrieval of similar past conversations for context.
    Supports per-user memory isolation using anonymous user IDs.
    """
    
    MEMORY_PATH = "memory_index"
    MEMORY_METADATA_FILE = "memory_metadata.json"
    USER_MEMORY_PATH = "user_memories"  # Directory for per-user session data
    
    def __init__(self, embeddings: Optional[CustomEmbeddings] = None):
        self.embeddings = embeddings or CustomEmbeddings(
            api_url=settings.EMBEDDING_API_URL,
            model=settings.EMBEDDING_MODEL,
            delay_between_requests=0.1,
        )
        self.vector_store: Optional[FAISS] = None
        self.metadata: Dict[str, Any] = {
            "conversations": {},
            "users": {},  # Track user metadata
        }
        self._initialize()
    
    def _initialize(self):
        """Initialize or load memory index"""
        # Ensure user memory directory exists
        os.makedirs(self.USER_MEMORY_PATH, exist_ok=True)
        
        if os.path.exists(self.MEMORY_PATH):
            try:
                self.vector_store = FAISS.load_local(
                    self.MEMORY_PATH,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                self._load_metadata()
                print(f"✓ Loaded memory index with {len(self.vector_store.index_to_docstore_id)} entries")
            except Exception as e:
                print(f"Could not load memory index: {e}. Creating new one.")
                self._create_empty_index()
        else:
            self._create_empty_index()
    
    def _create_empty_index(self):
        """Create an empty FAISS index for memory"""
        # Get embedding dimension
        test_embedding = self.embeddings.embed_query("test")
        dimension = len(test_embedding)
        
        # Create empty FAISS index
        index = faiss.IndexFlatL2(dimension)
        
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self.metadata = {
            "conversations": {},
            "users": {},
            "created_at": datetime.now().isoformat()
        }
        
        # Save empty index to disk immediately
        self.save()
        print("✓ Created new memory index")
    
    def _load_metadata(self):
        """Load metadata from file"""
        metadata_path = os.path.join(self.MEMORY_PATH, self.MEMORY_METADATA_FILE)
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
    
    def _save_metadata(self):
        """Save metadata to file"""
        os.makedirs(self.MEMORY_PATH, exist_ok=True)
        metadata_path = os.path.join(self.MEMORY_PATH, self.MEMORY_METADATA_FILE)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def store_conversation(
        self, 
        question: str, 
        answer: str, 
        sources: List[Dict] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Store a Q&A pair in memory.
        
        Args:
            question: User's question
            answer: AI's answer
            sources: Optional list of sources used
            user_id: Optional anonymous user ID
            session_id: Optional session ID for grouping conversations
        
        Returns:
            Conversation ID
        """
        conv_id = str(uuid4())
        timestamp = datetime.now().isoformat()
        
        # Create document combining Q&A for semantic search
        # Format: Question + Answer summary for better retrieval
        content = f"Question: {question}\n\nAnswer: {answer[:500]}"  # Truncate long answers
        
        doc = Document(
            page_content=content,
            metadata={
                "conv_id": conv_id,
                "question": question,
                "timestamp": timestamp,
                "type": "qa_pair",
                "user_id": user_id or "anonymous",
                "session_id": session_id,
            }
        )
        
        # Add to vector store
        self.vector_store.add_documents([doc], ids=[conv_id])
        
        # Store full conversation in metadata
        self.metadata["conversations"][conv_id] = {
            "question": question,
            "answer": answer,
            "sources": sources or [],
            "timestamp": timestamp,
            "user_id": user_id or "anonymous",
            "session_id": session_id,
        }
        
        # Track user metadata
        if user_id:
            if user_id not in self.metadata.get("users", {}):
                self.metadata["users"][user_id] = {
                    "first_seen": timestamp,
                    "conversation_count": 0,
                    "sessions": [],
                }
            self.metadata["users"][user_id]["conversation_count"] += 1
            self.metadata["users"][user_id]["last_seen"] = timestamp
            if session_id and session_id not in self.metadata["users"][user_id]["sessions"]:
                self.metadata["users"][user_id]["sessions"].append(session_id)
        
        # Also store in per-user session file for quick retrieval
        if user_id:
            self._store_user_session(user_id, session_id, conv_id, question, answer, timestamp)
        
        # Save to disk
        self.save()
        
        return conv_id
    
    def _store_user_session(
        self, 
        user_id: str, 
        session_id: Optional[str], 
        conv_id: str, 
        question: str, 
        answer: str, 
        timestamp: str
    ):
        """Store conversation in per-user session file for quick retrieval"""
        user_file = os.path.join(self.USER_MEMORY_PATH, f"{user_id}.json")
        
        user_data = {"sessions": {}, "conversations": []}
        if os.path.exists(user_file):
            try:
                with open(user_file, "r", encoding="utf-8") as f:
                    user_data = json.load(f)
            except Exception:
                pass
        
        # Store conversation
        conv_entry = {
            "id": conv_id,
            "question": question,
            "answer": answer,
            "timestamp": timestamp,
            "session_id": session_id,
        }
        user_data["conversations"].append(conv_entry)
        
        # Track session
        if session_id:
            if session_id not in user_data["sessions"]:
                user_data["sessions"][session_id] = {
                    "started": timestamp,
                    "conversations": [],
                }
            user_data["sessions"][session_id]["conversations"].append(conv_id)
            user_data["sessions"][session_id]["last_activity"] = timestamp
        
        # Limit stored conversations per user (keep last 100)
        if len(user_data["conversations"]) > 100:
            user_data["conversations"] = user_data["conversations"][-100:]
        
        with open(user_file, "w", encoding="utf-8") as f:
            json.dump(user_data, f, indent=2, ensure_ascii=False)
    
    def search_similar(
        self, 
        query: str, 
        k: int = 3, 
        user_id: Optional[str] = None,
        include_global: bool = True
    ) -> List[Dict]:
        """
        Search for similar past conversations.
        
        Args:
            query: Search query
            k: Number of results to return
            user_id: Optional user ID to filter results (prioritize user's own history)
            include_global: Whether to include global memory if user memory is insufficient
        
        Returns:
            List of similar past Q&A pairs with scores
        """
        if not self.vector_store or len(self.vector_store.index_to_docstore_id) == 0:
            return []
        
        try:
            # Get more results initially to filter
            fetch_k = k * 3 if user_id else k
            results = self.vector_store.similarity_search_with_score(query, k=fetch_k)
            
            similar_conversations = []
            user_conversations = []
            global_conversations = []
            
            for doc, score in results:
                conv_id = doc.metadata.get("conv_id")
                if conv_id and conv_id in self.metadata["conversations"]:
                    conv = self.metadata["conversations"][conv_id]
                    conv_entry = {
                        "question": conv["question"],
                        "answer": conv["answer"],
                        "score": float(score),
                        "timestamp": conv["timestamp"],
                        "user_id": conv.get("user_id", "anonymous"),
                    }
                    
                    # Separate user's own conversations from global
                    if user_id and conv.get("user_id") == user_id:
                        user_conversations.append(conv_entry)
                    else:
                        global_conversations.append(conv_entry)
            
            # Prioritize user's own history
            if user_id:
                similar_conversations.extend(user_conversations[:k])
                if include_global and len(similar_conversations) < k:
                    remaining = k - len(similar_conversations)
                    similar_conversations.extend(global_conversations[:remaining])
            else:
                similar_conversations = (user_conversations + global_conversations)[:k]
            
            return similar_conversations
        except Exception as e:
            print(f"Error searching memory: {e}")
            return []
    
    def get_user_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        """
        Get a user's conversation history.
        
        Args:
            user_id: The user's anonymous ID
            limit: Maximum number of conversations to return
        
        Returns:
            List of conversations, most recent first
        """
        user_file = os.path.join(self.USER_MEMORY_PATH, f"{user_id}.json")
        
        if not os.path.exists(user_file):
            return []
        
        try:
            with open(user_file, "r", encoding="utf-8") as f:
                user_data = json.load(f)
            
            conversations = user_data.get("conversations", [])
            # Return most recent first
            return conversations[-limit:][::-1]
        except Exception as e:
            print(f"Error loading user history: {e}")
            return []
    
    def get_session_history(self, user_id: str, session_id: str) -> List[Dict]:
        """
        Get conversation history for a specific session.
        
        Args:
            user_id: The user's anonymous ID
            session_id: The session ID
        
        Returns:
            List of conversations in chronological order
        """
        user_file = os.path.join(self.USER_MEMORY_PATH, f"{user_id}.json")
        
        if not os.path.exists(user_file):
            return []
        
        try:
            with open(user_file, "r", encoding="utf-8") as f:
                user_data = json.load(f)
            
            session_conv_ids = user_data.get("sessions", {}).get(session_id, {}).get("conversations", [])
            
            # Get conversations for this session
            session_conversations = [
                conv for conv in user_data.get("conversations", [])
                if conv.get("id") in session_conv_ids
            ]
            
            return session_conversations
        except Exception as e:
            print(f"Error loading session history: {e}")
            return []
    
    def get_memory_context(
        self, 
        query: str, 
        k: int = 2, 
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Get formatted memory context for a query.
        
        Args:
            query: Current query
            k: Number of similar conversations to include
            user_id: Optional user ID to prioritize user's history
            session_id: Optional session ID for recent context (enhanced for follow-ups)
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # For follow-ups, get recent session history first
        if session_id and user_id:
            recent = self.get_session_history(user_id, session_id)
            if recent:
                recent_items = recent[-3:]  # Last 3 conversations
                context_parts.append("Recent conversation in this session:")
                for i, conv in enumerate(recent_items, 1):
                    context_parts.append(
                        f"\n[Recent {i}]\n"
                        f"You asked: {conv.get('question', '')}\n"
                        f"I answered: {conv.get('answer', '')[:400]}..."
                    )
        
        # Also do semantic search
        similar = self.search_similar(query, k=k, user_id=user_id)
        
        if similar:
            if context_parts:
                context_parts.append("\nRelated past conversations:")
            else:
                context_parts.append("Previous relevant conversations:")
                
            for i, conv in enumerate(similar, 1):
                # Only include if score is good (lower is better in L2)
                if conv["score"] < 500:  # Threshold for relevance
                    is_own = " (from your previous chat)" if user_id and conv.get("user_id") == user_id else ""
                    context_parts.append(
                        f"\n[Past Q&A {i}{is_own}]\n"
                        f"User asked: {conv['question']}\n"
                        f"Answer: {conv['answer'][:300]}..."
                    )
        
        if not context_parts:
            return ""
        
        return "\n".join(context_parts)
    
    def save(self):
        """Save memory index and metadata to disk"""
        os.makedirs(self.MEMORY_PATH, exist_ok=True)
        self.vector_store.save_local(self.MEMORY_PATH)
        self._save_metadata()
    
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        return {
            "total_conversations": len(self.metadata.get("conversations", {})),
            "index_size": len(self.vector_store.index_to_docstore_id) if self.vector_store else 0,
            "total_users": len(self.metadata.get("users", {})),
        }
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get statistics for a specific user"""
        user_meta = self.metadata.get("users", {}).get(user_id, {})
        
        return {
            "user_id": user_id,
            "exists": bool(user_meta),
            "conversation_count": user_meta.get("conversation_count", 0),
            "first_seen": user_meta.get("first_seen"),
            "last_seen": user_meta.get("last_seen"),
            "session_count": len(user_meta.get("sessions", [])),
        }
    
    def clear_user_memory(self, user_id: str) -> bool:
        """
        Clear all memory for a specific user.
        
        Args:
            user_id: The user's anonymous ID
        
        Returns:
            True if successful
        """
        try:
            # Remove user file
            user_file = os.path.join(self.USER_MEMORY_PATH, f"{user_id}.json")
            if os.path.exists(user_file):
                os.remove(user_file)
            
            # Remove from metadata
            if user_id in self.metadata.get("users", {}):
                del self.metadata["users"][user_id]
            
            # Note: We don't remove from FAISS index as it would require rebuilding
            # The conversations will still exist but won't be linked to the user
            
            self._save_metadata()
            return True
        except Exception as e:
            print(f"Error clearing user memory: {e}")
            return False


# Singleton instance
_memory_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """Get or create memory service instance"""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service

