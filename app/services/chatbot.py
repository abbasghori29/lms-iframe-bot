"""
Chatbot service with RAG (Retrieval Augmented Generation)
Includes memory for past Q&A
"""
import os
import sys
from typing import List, Optional, Dict
from pathlib import Path

# Add project root to path to import train_vector_db
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings
from train_vector_db import CustomEmbeddings
import re

# Language detection
try:
    from langdetect import detect, LangDetectException
    LANG_SUPPORT = True
except ImportError:
    LANG_SUPPORT = False
    print("Warning: Language detection library not installed. Install langdetect for bilingual support.")

# Import memory services (Pinecone preferred, FAISS fallback)
if settings.PINECONE_API_KEY:
    from app.services.pinecone_memory import get_pinecone_memory_service as get_memory_service
    from app.services.pinecone_memory import PineconeMemoryService as MemoryService
    MEMORY_TYPE = "pinecone"
else:
    from app.services.memory import get_memory_service, MemoryService
    MEMORY_TYPE = "faiss"


class ChatbotService:
    """Chatbot service with RAG capabilities and memory"""
    
    def __init__(self):
        self.llm = None
        self.vector_store = None
        self.embeddings = None
        self.memory: Optional[MemoryService] = None
        self._initialize()
    
    def _initialize(self):
        """Initialize LLM and vector store"""
        # Initialize LLM with OpenAI as primary, Groq as fallback
        primary_llm = None
        fallback_llm = None
        
        # Try to initialize OpenAI as primary
        if settings.OPENAI_API_KEY:
            try:
                primary_llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.5,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                    api_key=settings.OPENAI_API_KEY,
                )
                print("✓ OpenAI LLM initialized as primary")
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI LLM: {e}")
        
        # Initialize Groq as fallback (or primary if OpenAI not available)
        if settings.GROQ_API_KEY:
            try:
                fallback_llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                    groq_api_key=settings.GROQ_API_KEY,
                )
                print("✓ Groq LLM initialized as fallback")
            except Exception as e:
                print(f"Warning: Could not initialize Groq LLM: {e}")
        
        # Set up LLM with fallback chain
        if primary_llm and fallback_llm:
            self.llm = primary_llm.with_fallbacks([fallback_llm])
            print("✓ LLM configured with OpenAI primary + Groq fallback")
        elif primary_llm:
            self.llm = primary_llm
            print("✓ Using OpenAI LLM only (no fallback)")
        elif fallback_llm:
            self.llm = fallback_llm
            print("✓ Using Groq LLM only (OpenAI not configured)")
        else:
            raise ValueError("No LLM API key configured. Set OPENAI_API_KEY or GROQ_API_KEY in .env")
        
        # Initialize embeddings
        self.embeddings = CustomEmbeddings(
            api_url=settings.EMBEDDING_API_URL,
            model=settings.EMBEDDING_MODEL,
            delay_between_requests=0.1,  # Faster for chat (fewer requests)
        )
        
        # Load vector store
        self._load_vector_store()
        
        # Initialize memory service (Pinecone if available, else FAISS)
        try:
            self.memory = get_memory_service(embeddings=self.embeddings)
            print(f"✓ Memory service initialized ({MEMORY_TYPE})")
        except Exception as e:
            print(f"Warning: Could not initialize memory service ({MEMORY_TYPE}): {e}")
    
    def _load_vector_store(self):
        """Load the FAISS vector store"""
        vector_store_path = settings.VECTOR_STORE_PATH
        
        if not os.path.exists(vector_store_path):
            raise FileNotFoundError(
                f"Vector store not found at {vector_store_path}. "
                "Please run train_vector_db.py first."
            )
        
        try:
            from langchain_community.vectorstores import FAISS
            self.vector_store = FAISS.load_local(
                vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print(f"✓ Loaded vector store with {len(self.vector_store.index_to_docstore_id)} documents")
        except Exception as e:
            raise Exception(f"Error loading vector store: {e}")
    
    def _create_prompt_template(self, include_memory: bool = False) -> ChatPromptTemplate:
        """Create the RAG prompt template"""
        system_prompt = """You are a helpful educational assistant for CAFS (Canadian Association of Financial Services).

FIRST: Check if the user's message is a greeting like "hi", "hello", "bonjour", "hey", "good morning", etc.
- If YES: Respond with a friendly greeting like "Hello! I'm here to help you with questions about CAFS courses, certifications, and training programs. What would you like to know?"
- If NO: Answer ONLY from the Context provided below. If the answer is not in the Context, say "This information is not available in our documentation."

Respond in the SAME LANGUAGE as the user's question.

CRITICAL: YOU MUST USE ONLY HTML TAGS - NO MARKDOWN ALLOWED

Your response MUST be formatted using ONLY these HTML tags:
- <h3>text</h3> for section headers
- <p>text</p> for paragraphs
- <ul><li>item</li></ul> for bullet lists
- <ol><li>item</li></ol> for numbered lists
- <strong>text</strong> for bold/emphasis

FORBIDDEN FORMATS (DO NOT USE):
- **bold** or __bold__ (use <strong> instead)
- # Heading (use <h3> instead)
- - bullet or * bullet (use <ul><li> instead)
- 1. numbered (use <ol><li> instead)
- Any markdown syntax

REQUIRED STRUCTURE:
1. Wrap ALL content in HTML tags
2. Use <p> for regular text paragraphs
3. Use <h3> for section titles
4. Use <ul> or <ol> with <li> for ANY list (never plain text lists)
5. Use <strong> for emphasis (never ** or __)

EXAMPLE - CORRECT FORMAT:
<h3>Course Requirements</h3>
<p>The following elements are required:</p>
<ol>
  <li><strong>Personal Circumstances:</strong> Understanding the client's situation.</li>
  <li><strong>Financial Circumstances:</strong> Knowing the client's financial state.</li>
</ol>
<p>For additional details:</p>
<ul>
  <li>Bullet point one</li>
  <li>Bullet point two</li>
</ul>

CRITICAL WARNING: When using <ol> for numbered lists, do NOT put numbers like "1." or "2." inside the <li> tags. The <ol> tag automatically adds numbers. 
WRONG: <li>1. First item</li>
CORRECT: <li>First item</li>

REMEMBER: If you use markdown (**, -, #, etc.), your response will be broken. ONLY use HTML tags.

CONTENT RULES:
- Professional tone.
- No document references (e.g. [Document 1]).

Context:
{context}
{memory_context}"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        
        return prompt
    
    def _normalize_markdown_to_html(self, text: str) -> str:
        """Convert any markdown that slipped through to HTML"""
        # First, convert markdown bold to HTML (do this before line processing)
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
        
        # Process line by line
        lines = text.split('\n')
        result_lines = []
        in_ul = False
        in_ol = False
        in_p = False
        current_paragraph = []
        
        for line in lines:
            stripped = line.strip()
            
            # Check for markdown heading
            heading_match = re.match(r'^#{1,6}\s+(.+)$', stripped)
            if heading_match:
                # Close any open tags
                if in_p:
                    result_lines.append(f'<p>{" ".join(current_paragraph)}</p>')
                    current_paragraph = []
                    in_p = False
                if in_ul:
                    result_lines.append('</ul>')
                    in_ul = False
                if in_ol:
                    result_lines.append('</ol>')
                    in_ol = False
                result_lines.append(f'<h3>{heading_match.group(1)}</h3>')
                continue
            
            # Check for unordered list item
            ul_match = re.match(r'^[-*]\s+(.+)$', stripped)
            if ul_match:
                if in_p:
                    result_lines.append(f'<p>{" ".join(current_paragraph)}</p>')
                    current_paragraph = []
                    in_p = False
                if in_ol:
                    result_lines.append('</ol>')
                    in_ol = False
                if not in_ul:
                    result_lines.append('<ul>')
                    in_ul = True
                result_lines.append(f'  <li>{ul_match.group(1)}</li>')
                continue
            
            # Check for ordered list item
            ol_match = re.match(r'^\d+\.\s+(.+)$', stripped)
            if ol_match:
                if in_p:
                    result_lines.append(f'<p>{" ".join(current_paragraph)}</p>')
                    current_paragraph = []
                    in_p = False
                if in_ul:
                    result_lines.append('</ul>')
                    in_ul = False
                if not in_ol:
                    result_lines.append('<ol>')
                    in_ol = True
                result_lines.append(f'  <li>{ol_match.group(1)}</li>')
                continue
            
            # Empty line - close open tags
            if not stripped:
                if in_p:
                    result_lines.append(f'<p>{" ".join(current_paragraph)}</p>')
                    current_paragraph = []
                    in_p = False
                if in_ul:
                    result_lines.append('</ul>')
                    in_ul = False
                if in_ol:
                    result_lines.append('</ol>')
                    in_ol = False
                continue
            
            # Check if line is already HTML
            if stripped.startswith('<') and ('</' in stripped or stripped.endswith('>')):
                # Close open tags
                if in_p:
                    result_lines.append(f'<p>{" ".join(current_paragraph)}</p>')
                    current_paragraph = []
                    in_p = False
                if in_ul:
                    result_lines.append('</ul>')
                    in_ul = False
                if in_ol:
                    result_lines.append('</ol>')
                    in_ol = False
                result_lines.append(line)
                continue
            
            # Regular text - add to paragraph
            if not in_p:
                in_p = True
            current_paragraph.append(stripped)
        
        # Close any remaining open tags
        if in_p:
            result_lines.append(f'<p>{" ".join(current_paragraph)}</p>')
        if in_ul:
            result_lines.append('</ul>')
        if in_ol:
            result_lines.append('</ol>')
        
        return '\n'.join(result_lines)
    
    def _clean_answer(self, answer: str) -> str:
        """Clean up the LLM answer - normalize to HTML and remove junk"""
        
        # Step 1: Remove document references
        doc_patterns = [
            r'【Document\s*\d+】\.?', r'\[Document\s*\d+\]\.?', r'\(Document\s*\d+\)\.?',
            r'Document\s*\d+:?', r'Source\s*\d+:?', r'\[Source:\s*[^\]]+\]',
            r'\(Source:\s*[^\)]+\)', r'Page\s*\d+:?', r'\[p\.\s*\d+\]', r'\(p\.\s*\d+\)',
        ]
        for pattern in doc_patterns:
            answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
        
        # Step 2: Normalize any markdown to HTML
        # Check if answer contains markdown patterns
        has_markdown = bool(re.search(r'(\*\*|__|^#{1,6}\s|^[-*]\s|^\d+\.\s)', answer, re.MULTILINE))
        if has_markdown:
            answer = self._normalize_markdown_to_html(answer)
        
        # Step 3: Ensure proper spacing after HTML block elements
        answer = answer.replace('</h3>', '</h3>\n')
        answer = answer.replace('</p>', '</p>\n')
        answer = answer.replace('</ul>', '</ul>\n')
        answer = answer.replace('</ol>', '</ol>\n')
        answer = answer.replace('</li>', '</li>\n')
        
        # Step 4: Clean up excessive whitespace
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        answer = answer.strip()
        
        return answer
    
    def _detect_language(self, text: str) -> str:
        """Detect the language of the input text"""
        if not LANG_SUPPORT:
            return "en"  # Default to English
        
        try:
            # Detect language
            lang = detect(text)
            # Normalize to 'en' or 'fr'
            if lang == 'fr':
                return 'fr'
            else:
                return 'en'  # Default to English for all other languages
        except (LangDetectException, Exception) as e:
            print(f"Language detection error: {e}, defaulting to English")
            return "en"
    
    def _translate_query(self, text: str, target_lang: str = "en") -> str:
        """Translate text to target language using LLM (for vector search)"""
        try:
            # Ensure LLM is initialized
            if not self.llm:
                print("LLM not initialized, cannot translate. Using original text.")
                return text
            
            # Detect source language
            source_lang = self._detect_language(text) if LANG_SUPPORT else "en"
            
            # If already in target language, return as is
            if source_lang == target_lang:
                return text
            
            # Use LLM to translate
            translation_prompt = f"""Translate the following text from {source_lang.upper()} to {target_lang.upper()}. 
Return ONLY the translation, nothing else. No explanations, no additional text, just the translated text.

Text to translate: {text}

Translation:"""
            
            response = self.llm.invoke(translation_prompt)
            
            # Extract translation from response
            if hasattr(response, 'content'):
                translated = response.content.strip()
            else:
                translated = str(response).strip()
            
            print(f"Translated query: '{text}' ({source_lang}) -> '{translated}' ({target_lang})")
            return translated
        except Exception as e:
            print(f"Translation error: {e}, using original text")
            return text
    
    def _format_context(self, docs: List) -> str:
        """Format retrieved documents as context"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            # Clean up context markers for cleaner output
            content = doc.page_content
            # Remove context markers if present
            content = content.replace("[Previous context:", "").replace("[Following context:", "")
            content = content.replace("]", "").strip()
            
            # Don't include document numbers in context to avoid LLM citing them
            context_parts.append(f"---\n{content}\n")
        
        return "\n".join(context_parts)
    
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
        Chat with the RAG-powered chatbot
        
        Args:
            question: User's question
            chat_history: List of previous messages in format [("human", "..."), ("ai", "...")]
            k: Number of documents to retrieve
            use_memory: Whether to search memory for similar past conversations
            store_in_memory: Whether to store this conversation in memory
            user_id: Anonymous user ID for personalized memory
            session_id: Session ID for grouping conversations
        
        Returns:
            Dictionary with answer, sources, and memory info
        """
        import time
        from concurrent.futures import ThreadPoolExecutor
        total_start = time.time()
        
        # Helper functions for parallel execution
        def do_vector_search():
            return self.vector_store.similarity_search(question, k=k)
        
        def do_memory_search():
            if use_memory and self.memory:
                return self.memory.get_memory_context(question, k=2, user_id=user_id)
            return ""
        
        # Step 1+2: Run vector search and memory search in PARALLEL
        step_start = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(do_vector_search)
            memory_future = executor.submit(do_memory_search)
            
            docs = vector_future.result()
            memory_context = memory_future.result()
        
        parallel_time = (time.time() - step_start) * 1000
        print(f"⏱️ [1+2] Vector + Memory search (PARALLEL): {parallel_time:.1f}ms")
        
        memory_used = bool(memory_context)
        
        # Step 3: Format context
        step_start = time.time()
        context = self._format_context(docs)
        print(f"⏱️ [3] Format context: {(time.time() - step_start)*1000:.1f}ms")
        
        # Step 4: Format chat history
        step_start = time.time()
        messages = []
        if chat_history:
            for role, content in chat_history:
                if role == "human":
                    messages.append(HumanMessage(content=content))
                elif role == "ai" or role == "assistant":
                    messages.append(AIMessage(content=content))
        print(f"⏱️ [4] Format chat history: {(time.time() - step_start)*1000:.1f}ms")
        
        # Step 5: Create prompt
        step_start = time.time()
        prompt = self._create_prompt_template(include_memory=memory_used)
        print(f"⏱️ [5] Create prompt: {(time.time() - step_start)*1000:.1f}ms")
        
        # Step 6: Create chain
        step_start = time.time()
        chain = (
            {
                "context": lambda x: context,
                "memory_context": lambda x: memory_context,
                "question": lambda x: x["question"],
                "chat_history": lambda x: messages,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        print(f"⏱️ [6] Create chain: {(time.time() - step_start)*1000:.1f}ms")
        
        # Step 7: LLM inference (THE BIG ONE)
        step_start = time.time()
        try:
            answer = chain.invoke({"question": question})
            llm_time = (time.time() - step_start)*1000
            print(f"⏱️ [7] LLM inference: {llm_time:.1f}ms ⭐ {'SLOW!' if llm_time > 3000 else ''}")
            
            # Step 8: Clean answer
            step_start = time.time()
            answer = self._clean_answer(answer)
            print(f"⏱️ [8] Clean answer: {(time.time() - step_start)*1000:.1f}ms")
        except Exception as e:
            error_msg = str(e)
            print(f"❌ LLM error after {(time.time() - step_start)*1000:.1f}ms: {error_msg}")
            if "model_decommissioned" in error_msg.lower() or "model" in error_msg.lower():
                error_msg = f"LLM model error: {error_msg}. Please check your API key and model availability."
            elif "api" in error_msg.lower() or "key" in error_msg.lower():
                error_msg = f"API error: {error_msg}. Please check your OpenAI/Groq API key configuration."
            
            return {
                "answer": f"I apologize, but I encountered an error while generating a response: {error_msg}",
                "sources": [],
                "context_used": 0,
                "quick_suggestions": [],
                "memory_used": False,
                "error": error_msg,
            }
        
        # Step 9: Extract sources
        step_start = time.time()
        sources = []
        for doc in docs:
            page_num = doc.metadata.get("page_number") or doc.metadata.get("page")
            if page_num is not None:
                page_str = str(page_num)
            else:
                page_str = "N/A"
            
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": page_str,
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            })
        print(f"⏱️ [9] Extract sources: {(time.time() - step_start)*1000:.1f}ms")
        
        # Step 10: Store in memory (BACKGROUND - non-blocking)
        def store_in_background():
            try:
                self.memory.store_conversation(
                    question, 
                    answer, 
                    sources,
                    user_id=user_id,
                    session_id=session_id
                )
                print(f"⏱️ [BG] Memory stored successfully")
            except Exception as e:
                print(f"Warning: Could not store in memory: {e}")
        
        if store_in_memory and self.memory and not any(x.get("error") for x in [{}]):
            import threading
            thread = threading.Thread(target=store_in_background, daemon=True)
            thread.start()
            print(f"⏱️ [10] Store in memory: BACKGROUND (non-blocking)")
        
        # Total time
        total_time = (time.time() - total_start)*1000
        print(f"⏱️ ═══════════════════════════════════════")
        print(f"⏱️ TOTAL TIME: {total_time:.1f}ms ({total_time/1000:.2f}s)")
        print(f"⏱️ ═══════════════════════════════════════")
        
        return {
            "answer": answer,
            "sources": sources,
            "context_used": len(docs),
            "quick_suggestions": [],
            "memory_used": memory_used,
            "user_id": user_id,
            "session_id": session_id,
        }


# Global chatbot instance (singleton)
_chatbot_service: Optional[ChatbotService] = None


def get_chatbot_service() -> ChatbotService:
    """Get or create chatbot service instance"""
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = ChatbotService()
    return _chatbot_service

