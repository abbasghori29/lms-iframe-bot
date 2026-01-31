"""
LangGraph-based Conversational RAG Service

This service solves the follow-up question problem by:
1. Contextualizing queries using chat history (rewriting "why?" into full questions)
2. Maintaining proper conversation state
3. Using the contextualized query for vector retrieval
4. Combining RAG context with conversation memory

Based on LangGraph best practices for handling conversation history and follow-ups.
"""
import os
from typing import List, Optional, Dict, Any, Annotated, Sequence, TypedDict
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from app.core.config import settings


class ConversationState(TypedDict):
    """State for the conversational RAG graph.
    
    This maintains all information needed across the conversation:
    - messages: Full conversation history (auto-managed by add_messages)
    - original_query: The user's original question
    - contextualized_query: Query rewritten with context for retrieval
    - is_followup: True if this is a follow-up question (has conversation history)
    - retrieved_docs: Documents retrieved from vector store
    - rag_context: Formatted context from retrieved documents
    - memory_context: Context from past conversations (Pinecone)
    - response: Final response to the user
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    original_query: str
    contextualized_query: str
    is_followup: bool  # True = has conversation history, LLM decides how to use it
    retrieved_docs: List[Document]
    rag_context: str
    memory_context: str
    response: str
    user_id: Optional[str]
    session_id: Optional[str]


class LangGraphRAGService:
    """
    LangGraph-based Conversational RAG Service.
    
    Key features:
    1. Query Contextualization: Rewrites follow-up questions into standalone queries
    2. Intelligent Retrieval: Uses contextualized query for better vector search
    3. Conversation Memory: Maintains full chat history for context
    4. State Management: Proper LangGraph state handling
    """
    
    def __init__(self):
        self.llm = None
        self.fast_llm = None  # For query rewriting (faster, cheaper)
        self.vector_store = None
        self.embeddings = None
        self.memory = None
        self.graph = None
        self._initialize()
    
    def _initialize(self):
        """Initialize all components"""
        self._setup_llms()
        self._setup_embeddings()
        self._load_vector_store()
        self._setup_memory()
        self._build_graph()
    
    def _setup_llms(self):
        """Setup LLMs with fallbacks"""
        primary_llm = None
        fallback_llm = None
        
        if settings.OPENAI_API_KEY:
            try:
                primary_llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.5,
                    api_key=settings.OPENAI_API_KEY,
                )
                # Use same model for fast operations (query rewriting)
                self.fast_llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    api_key=settings.OPENAI_API_KEY,
                )
                print("✓ OpenAI LLM initialized")
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI: {e}")
        
        if settings.GROQ_API_KEY:
            try:
                fallback_llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0,
                    groq_api_key=settings.GROQ_API_KEY,
                )
                if not self.fast_llm:
                    self.fast_llm = fallback_llm
                print("✓ Groq LLM initialized as fallback")
            except Exception as e:
                print(f"Warning: Could not initialize Groq: {e}")
        
        if primary_llm and fallback_llm:
            self.llm = primary_llm.with_fallbacks([fallback_llm])
        elif primary_llm:
            self.llm = primary_llm
        elif fallback_llm:
            self.llm = fallback_llm
        else:
            raise ValueError("No LLM configured")
        
        if not self.fast_llm:
            self.fast_llm = self.llm
    
    def _setup_embeddings(self):
        """Setup embeddings"""
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=settings.OPENAI_API_KEY,
        )
    
    def _load_vector_store(self):
        """Load FAISS vector store"""
        vector_store_path = settings.VECTOR_STORE_PATH
        if not os.path.exists(vector_store_path):
            raise FileNotFoundError(f"Vector store not found at {vector_store_path}")
        
        self.vector_store = FAISS.load_local(
            vector_store_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"✓ Vector store loaded with {len(self.vector_store.index_to_docstore_id)} documents")
    
    def _setup_memory(self):
        """Setup memory service"""
        if settings.PINECONE_API_KEY:
            from app.services.pinecone_memory import get_pinecone_memory_service
            self.memory = get_pinecone_memory_service(embeddings=self.embeddings)
            print("✓ Pinecone memory service initialized")
        else:
            from app.services.memory import get_memory_service
            self.memory = get_memory_service()
            print("✓ FAISS memory service initialized")
    
    def _build_graph(self):
        """Build the LangGraph conversation graph.
        
        Graph Structure:
        START -> contextualize_query -> retrieve -> generate -> END
        
        This ensures:
        1. Follow-up questions are rewritten before retrieval
        2. Retrieval uses the contextualized query
        3. Generation has access to full context
        """
        graph_builder = StateGraph(ConversationState)
        
        # Add nodes
        graph_builder.add_node("contextualize_query", self._contextualize_query_node)
        graph_builder.add_node("retrieve", self._retrieve_node)
        graph_builder.add_node("generate", self._generate_node)
        
        # Define edges (linear flow for now)
        graph_builder.add_edge(START, "contextualize_query")
        graph_builder.add_edge("contextualize_query", "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        graph_builder.add_edge("generate", END)
        
        # Compile the graph
        self.graph = graph_builder.compile()
        print("✓ LangGraph RAG pipeline built")
    
    def _contextualize_query_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Contextualize the user's query using conversation history.
        
        ALWAYS rewrites follow-up questions into standalone queries for better retrieval.
        The generate node will intelligently decide whether to use:
        - Retrieved context (if relevant)
        - Conversation history (if retrieval isn't helpful)
        - Both combined
        """
        original_query = state["original_query"]
        messages = state.get("messages", [])
        
        # If no history, it's a new question
        if not messages or len(messages) < 2:
            return {"contextualized_query": original_query, "is_followup": False}
        
        # Has history - this is a follow-up, always try to contextualize
        is_followup = True
        
        # Build contextualization prompt
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query rewriter. Your task is to reformulate follow-up questions 
into standalone questions that can be used for information retrieval.

RULES:
1. If the question is already clear and standalone, return it unchanged
2. If it's a follow-up (why?, explain more, what about X?), rewrite it to include the topic from history
3. Keep the same language as the original question
4. Keep the rewritten question concise but complete
5. DO NOT answer the question, just rewrite it
6. ALWAYS include the subject/topic being discussed

Examples:
- History: "What is CSI?" -> "CSI is a certification program for..." 
- Follow-up: "Why?" 
- Rewritten: "Why is CSI certification important?"

- History: "What is CSI?" -> "CSI is a certification program" 
- Follow-up: "Explain more" 
- Rewritten: "Explain more about CSI certification and its details"

- History: "Tell me about mutual funds" -> "Mutual funds are..."
- Follow-up: "What are the risks?"
- Rewritten: "What are the risks of investing in mutual funds?"

- History: "What is ETF?" -> "ETF is..."
- Follow-up: "How does it compare?"
- Rewritten: "How do ETFs compare to other investment options?"
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", """Rewrite this follow-up question as a standalone question for retrieval:
"{question}"

Standalone question:""")
        ])
        
        recent_messages = messages[-6:] if len(messages) > 6 else messages
        
        try:
            chain = contextualize_prompt | self.fast_llm | StrOutputParser()
            contextualized = chain.invoke({
                "chat_history": recent_messages,
                "question": original_query
            })
            contextualized = contextualized.strip().strip('"')
            
            if contextualized and len(contextualized) > 5:
                print(f"🔄 Query contextualized: '{original_query}' → '{contextualized}'")
                return {"contextualized_query": contextualized, "is_followup": is_followup}
        except Exception as e:
            print(f"Warning: Query contextualization failed: {e}")
        
        return {"contextualized_query": original_query, "is_followup": is_followup}
    
    def _retrieve_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        ALWAYS retrieve relevant documents using the contextualized query.
        
        The generate node will intelligently decide how to use:
        - Retrieved context (if relevant to user's intent)
        - Conversation history (if retrieval doesn't match intent)
        """
        query = state["contextualized_query"]
        user_id = state.get("user_id")
        session_id = state.get("session_id")
        
        # Always retrieve from vector store
        docs = self.vector_store.similarity_search(query, k=5)
        
        # Format RAG context
        context_parts = []
        for doc in docs:
            content = doc.page_content
            content = content.replace("[Previous context:", "").replace("[Following context:", "")
            content = content.replace("]", "").strip()
            context_parts.append(f"---\n{content}\n")
        
        rag_context = "\n".join(context_parts)
        
        # Get memory context
        memory_context = ""
        if self.memory:
            try:
                memory_context = self.memory.get_memory_context(
                    query, 
                    k=2, 
                    user_id=user_id,
                    session_id=session_id
                )
            except Exception as e:
                print(f"Warning: Memory retrieval failed: {e}")
        
        return {
            "retrieved_docs": docs,
            "rag_context": rag_context,
            "memory_context": memory_context
        }
    
    def _generate_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Generate the final response using RAG context and conversation history.
        
        The LLM intelligently decides:
        1. If retrieved context is relevant → Use it
        2. If retrieved context isn't relevant but it's a follow-up → Elaborate from conversation
        3. Combines both when appropriate
        """
        import re
        
        messages = state.get("messages", [])
        original_query = state["original_query"]
        rag_context = state["rag_context"]
        memory_context = state.get("memory_context", "")
        
        # Smart system prompt that lets LLM decide how to respond
        system_prompt = """You are a helpful educational assistant for CAFS (Canadian Association of Financial Services).

GREETING/CLOSING CHECK:
- Greetings (hi, hello, bonjour): Reply with a friendly greeting about CAFS
- Closing phrases (thanks, bye, merci): Reply with a friendly closing

FOR ALL QUESTIONS - INTELLIGENT RESPONSE STRATEGY:

You have access to TWO sources of information:
1. **Retrieved Context** (from knowledge base) - shown below
2. **Conversation History** (previous messages) - shown in chat history

DECISION LOGIC - Follow this carefully:

STEP 1: Check if Retrieved Context is RELEVANT to what user is asking
- Does it mention the topic they're asking about?
- Is the information related to their question?

STEP 2: Respond based on what you find:

A) IF Retrieved Context IS RELEVANT to their question:
   → Answer using the Retrieved Context
   → You can also reference conversation history for continuity

B) IF Retrieved Context is NOT RELEVANT but user is asking a FOLLOW-UP:
   (e.g., "why?", "explain more", "what do you mean?", "elaborate")
   → The user wants clarification on your PREVIOUS response
   → Look at your last response in conversation history
   → Elaborate, explain further, give examples, or rephrase
   → You don't need the Retrieved Context for this - use what you already said

C) IF Retrieved Context is NOT RELEVANT and it's a NEW topic:
   → Politely say you don't have information on this specific topic
   → Suggest what topics you CAN help with (based on CAFS/financial education)

IMPORTANT - NEVER DO THIS:
- Don't say "no information" when user just wants you to explain your previous answer more
- Don't ignore conversation history when user references something you just discussed
- Don't require exact topic matches - use related information when helpful

CRITICAL LANGUAGE REQUIREMENT - YOU MUST FOLLOW THIS STRICTLY:
- You MUST respond in the EXACT SAME LANGUAGE as the user's question
- If the user asks in English, you MUST respond entirely in English
- If the user asks in French, you MUST respond entirely in French
- DO NOT mix languages in your response
- DO NOT translate the user's question to another language
- Match the language of the user's question precisely

CRITICAL: USE ONLY HTML TAGS - NO MARKDOWN
- <h3>text</h3> for section headers
- <p>text</p> for paragraphs  
- <ul><li>item</li></ul> for bullet lists
- <ol><li>item</li></ol> for numbered lists
- <strong>text</strong> for bold/emphasis

Retrieved Context:
{context}
{memory_context}"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        # Use recent messages for chat history in prompt
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        
        try:
            response = chain.invoke({
                "context": rag_context,
                "memory_context": memory_context,
                "chat_history": recent_messages,
                "question": original_query
            })
            
            # Clean up the response (normalize markdown to HTML if needed)
            response = self._clean_response(response)
            
            return {"response": response}
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"❌ {error_msg}")
            return {"response": f"<p>I apologize, but I encountered an error: {error_msg}</p>"}
    
    def _clean_response(self, text: str) -> str:
        """Clean up the response, convert markdown to HTML if needed"""
        import re
        
        # Convert markdown bold to HTML
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
        
        # Convert markdown headers to HTML
        text = re.sub(r'^#{1,6}\s+(.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
        
        # Clean up spacing
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        return text
    
    def chat(
        self,
        question: str,
        chat_history: Optional[List] = None,
        k: int = 5,
        use_memory: bool = True,
        store_in_memory: bool = True,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Main chat interface - compatible with existing ChatbotService API.
        
        Args:
            question: User's question
            chat_history: List of previous messages [("human", "..."), ("ai", "...")]
            k: Number of documents to retrieve
            use_memory: Whether to search memory
            store_in_memory: Whether to store conversation
            user_id: User ID for personalization
            session_id: Session ID for grouping
        
        Returns:
            Dictionary with answer, sources, etc.
        """
        import time
        total_start = time.time()
        
        # Convert chat history to LangChain messages
        messages = []
        if chat_history:
            for role, content in chat_history:
                if role == "human":
                    messages.append(HumanMessage(content=content))
                elif role in ("ai", "assistant"):
                    messages.append(AIMessage(content=content))
        
        # Create initial state
        initial_state: ConversationState = {
            "messages": messages,
            "original_query": question,
            "contextualized_query": "",
            "is_followup": False,  # Will be set by contextualize node
            "retrieved_docs": [],
            "rag_context": "",
            "memory_context": "" if not use_memory else "",
            "response": "",
            "user_id": user_id,
            "session_id": session_id,
        }
        
        # Run the graph
        try:
            step_start = time.time()
            final_state = self.graph.invoke(initial_state)
            graph_time = (time.time() - step_start) * 1000
            print(f"⏱️ LangGraph execution: {graph_time:.1f}ms")
        except Exception as e:
            print(f"❌ Graph execution error: {e}")
            return {
                "answer": f"<p>I apologize, but I encountered an error: {str(e)}</p>",
                "sources": [],
                "context_used": 0,
                "quick_suggestions": [],
                "memory_used": False,
                "error": str(e),
            }
        
        # Extract results
        answer = final_state.get("response", "")
        docs = final_state.get("retrieved_docs", [])
        memory_used = bool(final_state.get("memory_context"))
        
        # Extract sources
        sources = []
        for doc in docs:
            page_num = doc.metadata.get("page_number") or doc.metadata.get("page")
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": str(page_num) if page_num else "N/A",
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            })
        
        # Store in memory (background)
        if store_in_memory and self.memory and answer:
            import threading
            def store_bg():
                try:
                    self.memory.store_conversation(
                        question, answer, sources,
                        user_id=user_id, session_id=session_id
                    )
                except Exception as e:
                    print(f"Warning: Memory storage failed: {e}")
            
            thread = threading.Thread(target=store_bg, daemon=True)
            thread.start()
        
        total_time = (time.time() - total_start) * 1000
        print(f"⏱️ Total LangGraph RAG time: {total_time:.1f}ms")
        
        return {
            "answer": answer,
            "sources": sources,
            "context_used": len(docs),
            "quick_suggestions": [],
            "memory_used": memory_used,
            "user_id": user_id,
            "session_id": session_id,
            "contextualized_query": final_state.get("contextualized_query", question),
        }


# Singleton instance
_langgraph_service: Optional[LangGraphRAGService] = None


def get_langgraph_service() -> LangGraphRAGService:
    """Get or create the LangGraph RAG service"""
    global _langgraph_service
    if _langgraph_service is None:
        _langgraph_service = LangGraphRAGService()
    return _langgraph_service

