"""
LangGraph-based Conversational RAG Service (Multimodal)

This service solves the follow-up question problem by:
1. Analyzing images with GPT-4o vision (when present)
2. Contextualizing queries using chat history (rewriting "why?" into full questions)
3. Maintaining proper conversation state
4. Using the contextualized query for vector retrieval
5. Combining RAG context + image understanding + conversation memory

Based on LangGraph best practices for handling conversation history, follow-ups, and multimodal inputs.
"""
import os
import asyncio
import base64
import json
from typing import AsyncGenerator, List, Optional, Dict, Any, Annotated, Sequence, TypedDict

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
    - image_data: Base64-encoded image data (if user sent an image)
    - image_mime_type: MIME type of the image (image/png, image/jpeg, etc.)
    - image_analysis: GPT-4o vision analysis of the image
    - has_image: Whether this turn includes an image
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
    # Multimodal fields
    image_data: Optional[str]
    image_mime_type: Optional[str]
    image_analysis: str
    has_image: bool


class LangGraphRAGService:
    """
    LangGraph-based Conversational RAG Service with Multimodal Support.
    
    Key features:
    1. Image Analysis: GPT-4o vision understands images, extracts text, identifies topics
    2. Query Contextualization: Rewrites follow-up questions into standalone queries
    3. Intelligent Retrieval: Uses contextualized query (enriched by image) for vector search
    4. Conversation Memory: Maintains full chat history for context
    5. Vision-Aware Generation: GPT-4o generates responses seeing the image + RAG context
    """
    
    def __init__(self):
        self.llm = None
        self.fast_llm = None  # For query rewriting (faster, cheaper)
        self.vision_llm = None  # GPT-4o for image understanding
        self.vector_store = None
        self.embeddings = None
        self.memory = None
        self.graph = None
        # Limit concurrent LLM invocations so the server doesn't get
        # overwhelmed. 50 lets ~100 HTTP connections queue safely while
        # 50 are actively waiting on OpenAI/Groq.
        self._semaphore: Optional[asyncio.Semaphore] = None
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
                # GPT-4o for vision — best-in-class image understanding
                self.vision_llm = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0.3,
                    api_key=settings.OPENAI_API_KEY,
                    max_tokens=4096,
                )
                print("✓ OpenAI LLMs initialized (gpt-4o-mini + gpt-4o vision)")
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
        
        # Vision LLM falls back to primary if not set (gpt-4o-mini has basic vision)
        if not self.vision_llm:
            self.vision_llm = primary_llm or self.llm
    
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
        START -> analyze_image -> contextualize_query -> retrieve -> generate -> END
        
        This ensures:
        1. Images are understood before any text processing
        2. Follow-up questions are rewritten before retrieval
        3. Retrieval uses the contextualized + image-enriched query
        4. Generation has access to image + RAG context + conversation history
        """
        graph_builder = StateGraph(ConversationState)
        
        # Add nodes
        graph_builder.add_node("analyze_image", self._analyze_image_node)
        graph_builder.add_node("contextualize_query", self._contextualize_query_node)
        graph_builder.add_node("retrieve", self._retrieve_node)
        graph_builder.add_node("generate", self._generate_node)
        
        # Define edges: image analysis → contextualize → retrieve → generate
        graph_builder.add_edge(START, "analyze_image")
        graph_builder.add_edge("analyze_image", "contextualize_query")
        graph_builder.add_edge("contextualize_query", "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        graph_builder.add_edge("generate", END)
        
        # Compile the graph
        self.graph = graph_builder.compile()
        print("✓ LangGraph RAG pipeline built (multimodal)")
    
    async def _analyze_image_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Analyze user-uploaded image using GPT-4o vision (async).
        
        This node:
        1. Describes the image content in detail
        2. Extracts any visible text (OCR)
        3. Identifies the subject/topic
        4. Generates a knowledge-base search query based on image + user question
        
        If no image is present, this is a no-op pass-through.
        """
        image_data = state.get("image_data")
        image_mime_type = state.get("image_mime_type", "image/png")
        original_query = state.get("original_query", "")
        
        if not image_data:
            return {"image_analysis": "", "has_image": False}
        
        print("🖼️ Analyzing image with GPT-4o vision...")
        
        # Build the user question context
        question_context = f"User's accompanying question: \"{original_query}\"" if original_query else "No accompanying question — analyze the image fully."
        
        # Create multimodal message for GPT-4o
        analysis_message = HumanMessage(content=[
            {
                "type": "text",
                "text": f"""You are an expert image analyst for a financial education platform (CAFS — Canadian Association of Financial Services).

Analyze this image thoroughly and provide:

1. **DESCRIPTION**: What is shown in the image? Be detailed — charts, tables, diagrams, screenshots, documents, formulas, etc.
2. **TEXT_FOUND**: Extract ALL visible text in the image (OCR). If it's a document/slide, capture the full content.
3. **TOPIC**: What financial/educational topic does this relate to? (e.g., mutual funds, KYC rules, CSI certification, ETFs, portfolio management, regulations, etc.)
4. **SEARCH_QUERY**: Based on the image content and the user's question, write a clear search query to find relevant information in our CAFS financial education knowledge base.

{question_context}

Respond in this exact format:
DESCRIPTION: [your description]
TEXT_FOUND: [extracted text or "No readable text found"]
TOPIC: [identified topic]
SEARCH_QUERY: [search query for knowledge base]"""
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_mime_type};base64,{image_data}",
                    "detail": "high"  # High detail for better OCR and analysis
                }
            }
        ])
        
        try:
            response = await self.vision_llm.ainvoke([analysis_message])
            analysis = response.content.strip()
            
            print(f"🖼️ Image analysis complete ({len(analysis)} chars)")
            
            # Extract search query from the structured response
            search_query = ""
            for line in analysis.split("\n"):
                if line.strip().upper().startswith("SEARCH_QUERY:"):
                    search_query = line.split(":", 1)[1].strip()
                    break
            
            if search_query:
                print(f"🔍 Image-derived search query: '{search_query}'")
            
            return {
                "image_analysis": analysis,
                "has_image": True,
            }
        except Exception as e:
            print(f"⚠️ Image analysis failed: {e}")
            # Graceful fallback — continue without image analysis
            return {
                "image_analysis": f"[Image was provided but analysis failed: {str(e)}]",
                "has_image": True,
            }
    
    async def _contextualize_query_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Intelligently build the best possible retrieval query using the LLM.

        Three scenarios — every one uses the LLM for query crafting:

        1. **Image present (with or without text):**
           The LLM receives the full image analysis (description, OCR text,
           topic, suggested search query) plus the user's text (if any) plus
           conversation history and produces one sharp, optimised retrieval
           query.  No string concatenation, no hardcoded phrase lists.

        2. **Text-only follow-up (has chat history):**
           The LLM rewrites the follow-up into a standalone question that
           includes the topic from history.

        3. **Text-only first message (no history):**
           The query is already standalone — pass through directly.
        """
        original_query = state["original_query"]
        messages = state.get("messages", [])
        image_analysis = state.get("image_analysis", "")
        has_image = state.get("has_image", False)
        is_followup = len(messages) >= 2

        # ─────────────────────────────────────────────────────────────────
        # SCENARIO 1: Image present — LLM crafts the retrieval query
        # ─────────────────────────────────────────────────────────────────
        if has_image and image_analysis:
            image_query_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a search-query engineer for a financial education knowledge base (CAFS — Canadian Association of Financial Services).

You will receive:
- An IMAGE ANALYSIS produced by a vision model (contains a description, extracted text, identified topic, and a suggested search query).
- The USER'S TEXT message (may be empty if they only sent an image).
- Recent CONVERSATION HISTORY (may be empty).

YOUR TASK:
Combine ALL of this information into ONE precise, comprehensive search query that will retrieve the most relevant documents from the knowledge base.

RULES:
1. The query must capture the core topic and specific details from the image.
2. If the user asked a specific question, the query must reflect that intent (e.g. "compare", "explain risks", "what are the requirements").
3. If conversation history exists, incorporate the ongoing topic so context is not lost.
4. Output ONLY the search query — nothing else. No explanations, no labels, no quotes.
5. Keep the query concise but information-rich (1-3 sentences max).
6. Use the same language as the user's text (default English)."""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", """IMAGE ANALYSIS:
{image_analysis}

USER'S TEXT: {user_text}

Search query:""")
            ])

            recent_messages = messages[-6:] if len(messages) > 6 else messages

            try:
                chain = image_query_prompt | self.fast_llm | StrOutputParser()
                crafted_query = await chain.ainvoke({
                    "chat_history": recent_messages,
                    "image_analysis": image_analysis,
                    "user_text": original_query if original_query else "(no text — image only)",
                })
                crafted_query = crafted_query.strip().strip('"')

                if crafted_query and len(crafted_query) > 5:
                    print(f"🖼️ LLM-crafted image retrieval query: '{crafted_query}'")
                    return {"contextualized_query": crafted_query, "is_followup": is_followup}
            except Exception as e:
                print(f"Warning: Image query crafting failed: {e}")

            # Fallback — extract SEARCH_QUERY or TOPIC from the analysis directly
            fallback = ""
            for line in image_analysis.split("\n"):
                stripped = line.strip()
                if stripped.upper().startswith("SEARCH_QUERY:"):
                    fallback = stripped.split(":", 1)[1].strip()
                    break
                elif stripped.upper().startswith("TOPIC:") and not fallback:
                    fallback = stripped.split(":", 1)[1].strip()

            result = fallback or image_analysis[:200]
            if original_query and original_query.strip():
                result = f"{original_query.strip()} {result}"
            print(f"🖼️ Fallback image query: '{result}'")
            return {"contextualized_query": result, "is_followup": is_followup}

        # ─────────────────────────────────────────────────────────────────
        # SCENARIO 2: Text-only, first message — no rewriting needed
        # ─────────────────────────────────────────────────────────────────
        if not is_followup:
            return {"contextualized_query": original_query, "is_followup": False}

        # ─────────────────────────────────────────────────────────────────
        # SCENARIO 3: Text-only follow-up — rewrite with chat context
        # ─────────────────────────────────────────────────────────────────
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
            contextualized = await chain.ainvoke({
                "chat_history": recent_messages,
                "question": original_query
            })
            contextualized = contextualized.strip().strip('"')

            if contextualized and len(contextualized) > 5:
                print(f"🔄 Query contextualized: '{original_query}' → '{contextualized}'")
                return {"contextualized_query": contextualized, "is_followup": True}
        except Exception as e:
            print(f"Warning: Query contextualization failed: {e}")

        return {"contextualized_query": original_query, "is_followup": True}
    
    async def _retrieve_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Retrieve relevant documents using the contextualized query (async).
        
        FAISS vector search and Pinecone memory search run **in parallel**
        via asyncio.gather(), saving 300-900ms per request.
        """
        query = state["contextualized_query"]
        user_id = state.get("user_id")
        session_id = state.get("session_id")

        # ── Launch FAISS + memory in parallel ──────────────────────────────
        async def _faiss_search() -> List[Document]:
            """FAISS similarity search — return top-k and let the LLM decide relevance."""
            loop = asyncio.get_event_loop()
            results_with_scores = await loop.run_in_executor(
                None, lambda: self.vector_store.similarity_search_with_score(query, k=5)
            )
            # Log scores for debugging (L2 distance — lower = better)
            if results_with_scores:
                scores = [f"{s:.3f}" for _, s in results_with_scores]
                print(f"📊 FAISS: {len(results_with_scores)} docs retrieved (L2 scores: {', '.join(scores)})")
            else:
                print("📊 FAISS: no results found")
            return [doc for doc, _ in results_with_scores]

        async def _memory_search() -> str:
            """Pinecone/FAISS memory — uses the async method we added."""
            if not self.memory:
                return ""
            try:
                return await self.memory.aget_memory_context(
                    query, k=2, user_id=user_id, session_id=session_id,
                )
            except Exception as e:
                print(f"Warning: Memory retrieval failed: {e}")
                return ""

        docs, memory_context = await asyncio.gather(
            _faiss_search(), _memory_search()
        )

        # Format RAG context
        context_parts = []
        for doc in docs:
            content = doc.page_content
            content = content.replace("[Previous context:", "").replace("[Following context:", "")
            content = content.replace("]", "").strip()
            context_parts.append(f"---\n{content}\n")
        
        rag_context = "\n".join(context_parts)
        
        return {
            "retrieved_docs": docs,
            "rag_context": rag_context,
            "memory_context": memory_context
        }
    
    async def _generate_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Generate the final response using RAG context, conversation history, and image (async).
        
        When an image is present:
        - Uses GPT-4o (vision) so it can SEE the actual image
        - Includes image analysis for additional context
        - Combines image understanding with RAG-retrieved knowledge
        
        The LLM intelligently decides:
        1. If retrieved context is relevant → Use it
        2. If retrieved context isn't relevant but it's a follow-up → Elaborate from conversation
        3. Combines both when appropriate
        4. When image is present → Reference what it sees in the image
        """
        import re
        
        messages = state.get("messages", [])
        original_query = state["original_query"]
        rag_context = state["rag_context"]
        memory_context = state.get("memory_context", "")
        has_image = state.get("has_image", False)
        image_data = state.get("image_data")
        image_mime_type = state.get("image_mime_type", "image/png")
        image_analysis = state.get("image_analysis", "")
        
        # Build the system prompt — includes image analysis when available
        image_instructions = ""
        if has_image and image_analysis:
            image_instructions = f"""

IMAGE ANALYSIS (from vision model):
{image_analysis}

CRITICAL IMAGE + KNOWLEDGE BASE INSTRUCTIONS:
You MUST follow ALL of these rules when an image is present:

1. You can SEE the image — reference specific visual details (charts, tables, text, diagrams).
2. You MUST incorporate and cite information from the Retrieved Context below.
   - The Retrieved Context comes from our CAFS knowledge base and contains authoritative course material.
   - DO NOT just describe the image from your own knowledge — that is NOT enough.
   - Your response MUST blend what you see in the image WITH the retrieved knowledge base content.
   - Quote or paraphrase specific facts, definitions, and explanations from the Retrieved Context.
3. If the image shows a concept (e.g. behavioral finance, risk profiling), use the Retrieved Context
   to provide the CAFS-specific explanation, terminology, and details — not generic descriptions.
4. If the image contains text, quote the relevant parts AND supplement with Retrieved Context.
5. Structure: First briefly describe what the image shows, THEN provide the in-depth explanation
   using the knowledge base content, connecting it back to the image throughout.
6. If Retrieved Context is empty or truly unrelated, still describe the image but explicitly note
   that you are answering from the image alone without knowledge base support."""
        
        system_prompt = """You are a helpful educational assistant for CAFS (Canadian Association of Financial Services).

GREETING/CLOSING CHECK:
- Greetings (hi, hello, bonjour): Reply with a friendly greeting about CAFS
- Closing phrases (thanks, bye, merci): Reply with a friendly closing

FOR ALL QUESTIONS - INTELLIGENT RESPONSE STRATEGY:

You have access to these sources of information:
1. **Retrieved Context** (from knowledge base) - shown below
2. **Conversation History** (previous messages) - shown in chat history
3. **Image** (if provided by the user) - you can see it directly
{image_instructions}

DECISION LOGIC - Follow this carefully:

STEP 1: Check if Retrieved Context is RELEVANT to what user is asking
- Does it mention the topic they're asking about?
- Is the information related to their question or image?

STEP 2: Respond based on what you find:

A) IF Retrieved Context IS RELEVANT to their question:
   → Answer using the Retrieved Context as your PRIMARY source of information
   → If an image is present, you MUST weave together the image content AND the retrieved knowledge
   → Quote or paraphrase specific passages from the Retrieved Context — don't just summarize
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
        
        # Use recent messages for chat history in prompt
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        
        try:
            if has_image and image_data:
                # ─── Multimodal generation: GPT-4o sees the image ───
                # Build messages manually to include the image
                formatted_system = system_prompt.format(
                    image_instructions=image_instructions,
                    context=rag_context,
                    memory_context=memory_context,
                )
                
                llm_messages: list[BaseMessage] = [SystemMessage(content=formatted_system)]
                
                # Add conversation history
                for msg in recent_messages:
                    llm_messages.append(msg)
                
                # Add the user's current message with the image
                human_content = []
                if original_query:
                    human_content.append({"type": "text", "text": original_query})
                human_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_mime_type};base64,{image_data}",
                        "detail": "high"
                    }
                })
                llm_messages.append(HumanMessage(content=human_content))
                
                # Use vision LLM for image-aware generation (async)
                response_msg = await self.vision_llm.ainvoke(llm_messages)
                response = response_msg.content
                
            else:
                # ─── Text-only generation (original flow, async) ───
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}"),
                ])
                
                chain = prompt | self.llm | StrOutputParser()
                
                response = await chain.ainvoke({
                    "image_instructions": image_instructions,
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
        image_data: Optional[str] = None,
        image_mime_type: Optional[str] = None,
    ) -> dict:
        """
        Main chat interface — supports text, images, or both.
        
        Args:
            question: User's question (can be empty if image-only)
            chat_history: List of previous messages [("human", "..."), ("ai", "...")]
            k: Number of documents to retrieve
            use_memory: Whether to search memory
            store_in_memory: Whether to store conversation
            user_id: User ID for personalization
            session_id: Session ID for grouping
            image_data: Base64-encoded image (optional)
            image_mime_type: MIME type of image, e.g. "image/png" (optional)
        
        Returns:
            Dictionary with answer, sources, etc.
        """
        import time
        total_start = time.time()
        
        has_image = bool(image_data)
        if has_image:
            print(f"🖼️ Image received ({image_mime_type or 'unknown type'}, {len(image_data) // 1024}KB base64)")
        
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
            "original_query": question or "",
            "contextualized_query": "",
            "is_followup": False,
            "retrieved_docs": [],
            "rag_context": "",
            "memory_context": "" if not use_memory else "",
            "response": "",
            "user_id": user_id,
            "session_id": session_id,
            # Multimodal
            "image_data": image_data,
            "image_mime_type": image_mime_type or "image/png",
            "image_analysis": "",
            "has_image": has_image,
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
        
        # Store in memory (background) — store image description too
        if store_in_memory and self.memory and answer:
            import threading
            memory_question = question
            if has_image and final_state.get("image_analysis"):
                # Include image context in memory for future retrieval
                memory_question = f"{question} [Image: {final_state['image_analysis'][:200]}]"
            
            def store_bg():
                try:
                    self.memory.store_conversation(
                        memory_question, answer, sources,
                        user_id=user_id, session_id=session_id
                    )
                except Exception as e:
                    print(f"Warning: Memory storage failed: {e}")
            
            thread = threading.Thread(target=store_bg, daemon=True)
            thread.start()
        
        total_time = (time.time() - total_start) * 1000
        print(f"⏱️ Total LangGraph RAG time: {total_time:.1f}ms {'(with image)' if has_image else ''}")
        
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


    async def astream_chat(
        self,
        question: str,
        chat_history: Optional[List] = None,
        k: int = 5,
        use_memory: bool = True,
        store_in_memory: bool = True,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        image_data: Optional[str] = None,
        image_mime_type: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream the response token-by-token as Server-Sent Events (SSE).

        Yields SSE-formatted strings:
          data: {"type": "token",  "content": "Hello"}
          data: {"type": "token",  "content": " world"}
          data: {"type": "done",   "sources": [...], "memory_used": true}
          data: [DONE]

        The frontend reads this with fetch() + ReadableStream so the user
        sees words appear character-by-character (like ChatGPT) instead of
        waiting the full 5-15s for a complete response.
        """
        import time
        total_start = time.time()
        has_image = bool(image_data)

        # Build message history
        messages = []
        if chat_history:
            for role, content in chat_history:
                if role == "human":
                    messages.append(HumanMessage(content=content))
                elif role in ("ai", "assistant"):
                    messages.append(AIMessage(content=content))

        initial_state: ConversationState = {
            "messages": messages,
            "original_query": question or "",
            "contextualized_query": "",
            "is_followup": False,
            "retrieved_docs": [],
            "rag_context": "",
            "memory_context": "",
            "response": "",
            "user_id": user_id,
            "session_id": session_id,
            "image_data": image_data,
            "image_mime_type": image_mime_type or "image/png",
            "image_analysis": "",
            "has_image": has_image,
        }

        retrieved_docs: List[Document] = []
        memory_context = ""
        image_analysis_text = ""
        full_response_parts: List[str] = []
        contextualized_query = question or ""

        try:
            async with self._get_semaphore():
                async for event in self.graph.astream_events(initial_state, version="v2"):
                    kind = event["event"]

                    # Safely extract node name — metadata can be str, dict, or missing
                    metadata = event.get("metadata")
                    node = metadata.get("langgraph_node", "") if isinstance(metadata, dict) else ""

                    data = event.get("data")
                    if not isinstance(data, dict):
                        continue  # skip events with non-dict data

                    # ── Stream tokens only from the final generate node ────────
                    if kind == "on_chat_model_stream" and node == "generate":
                        chunk = data.get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            full_response_parts.append(chunk.content)
                            yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

                    # ── Capture retrieved docs once retrieve node finishes ─────
                    elif kind == "on_chain_end" and node == "retrieve":
                        out = data.get("output")
                        if isinstance(out, dict):
                            retrieved_docs = out.get("retrieved_docs", [])
                            memory_context = out.get("memory_context", "")

                    # ── Capture contextualized query ───────────────────────────
                    elif kind == "on_chain_end" and node == "contextualize_query":
                        out = data.get("output")
                        if isinstance(out, dict):
                            contextualized_query = out.get("contextualized_query", question or "")
                        elif isinstance(out, str) and out.strip():
                            contextualized_query = out.strip()

                    # ── Capture image analysis for memory storage ──────────────
                    elif kind == "on_chain_end" and node == "analyze_image":
                        out = data.get("output")
                        if isinstance(out, dict):
                            image_analysis_text = out.get("image_analysis", "")

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Build sources list
        sources = []
        for doc in retrieved_docs:
            page_num = doc.metadata.get("page_number") or doc.metadata.get("page")
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": str(page_num) if page_num else "N/A",
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            })

        # Send the final metadata event so frontend knows we're done
        yield f"data: {json.dumps({'type': 'done', 'sources': sources, 'memory_used': bool(memory_context), 'contextualized_query': contextualized_query})}\n\n"
        yield "data: [DONE]\n\n"

        # Store in memory (fire-and-forget, non-blocking)
        answer = "".join(full_response_parts)
        if store_in_memory and self.memory and answer:
            memory_question = question or ""
            if has_image:
                # Include image analysis summary for better future memory recall
                img_summary = ""
                for line in (image_analysis_text or "").split("\n"):
                    if line.strip().upper().startswith("TOPIC:"):
                        img_summary = line.split(":", 1)[1].strip()
                        break
                memory_question = f"{question} [Image: {img_summary or 'image provided'}]"

            async def _store_stream_memory() -> None:
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        lambda: self.memory.store_conversation(
                            memory_question, answer, sources,
                            user_id=user_id, session_id=session_id,
                        ),
                    )
                except Exception as exc:
                    print(f"Warning: Stream memory storage failed: {exc}")

            asyncio.create_task(_store_stream_memory())

        total_time = (time.time() - total_start) * 1000
        print(f"⏱️ Total stream time: {total_time:.1f}ms")

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Return (lazily created) semaphore — must be called inside a running event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(50)  # max 50 concurrent LLM calls
        return self._semaphore

    async def achat(
        self,
        question: str,
        chat_history: Optional[List] = None,
        k: int = 5,
        use_memory: bool = True,
        store_in_memory: bool = True,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        image_data: Optional[str] = None,
        image_mime_type: Optional[str] = None,
    ) -> dict:
        """
        Async version of chat() — uses graph.ainvoke() so the event loop is
        never blocked while waiting for OpenAI / Groq responses.

        This is the method that MUST be called from async FastAPI endpoints.
        It allows hundreds of concurrent requests to share the same Uvicorn
        worker without stalling each other.
        """
        import time
        total_start = time.time()

        has_image = bool(image_data)
        if has_image:
            print(f"🖼️ Image received ({image_mime_type or 'unknown type'}, {len(image_data) // 1024}KB base64)")

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
            "original_query": question or "",
            "contextualized_query": "",
            "is_followup": False,
            "retrieved_docs": [],
            "rag_context": "",
            "memory_context": "",
            "response": "",
            "user_id": user_id,
            "session_id": session_id,
            # Multimodal
            "image_data": image_data,
            "image_mime_type": image_mime_type or "image/png",
            "image_analysis": "",
            "has_image": has_image,
        }

        # ── Run the graph asynchronously ──────────────────────────────────────
        # The semaphore caps concurrent LLM calls so we don't spam the API or
        # exhaust memory under burst traffic.
        try:
            async with self._get_semaphore():
                step_start = time.time()
                final_state = await self.graph.ainvoke(initial_state)
                graph_time = (time.time() - step_start) * 1000
                print(f"⏱️ LangGraph async execution: {graph_time:.1f}ms")
        except Exception as e:
            print(f"❌ Async graph execution error: {e}")
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

        # Build sources list
        sources = []
        for doc in docs:
            page_num = doc.metadata.get("page_number") or doc.metadata.get("page")
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": str(page_num) if page_num else "N/A",
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            })

        # ── Store memory as a fire-and-forget async task ──────────────────────
        # asyncio.create_task() schedules the coroutine on the running event
        # loop without blocking the response — replaces the old threading.Thread.
        if store_in_memory and self.memory and answer:
            memory_question = question
            if has_image and final_state.get("image_analysis"):
                memory_question = f"{question} [Image: {final_state['image_analysis'][:200]}]"

            async def _store_memory() -> None:
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        lambda: self.memory.store_conversation(
                            memory_question, answer, sources,
                            user_id=user_id, session_id=session_id,
                        ),
                    )
                except Exception as exc:
                    print(f"Warning: Async memory storage failed: {exc}")

            asyncio.create_task(_store_memory())

        total_time = (time.time() - total_start) * 1000
        print(f"⏱️ Total async RAG time: {total_time:.1f}ms {'(with image)' if has_image else ''}")

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
