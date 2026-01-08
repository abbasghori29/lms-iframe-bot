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
        # Initialize Groq LLM
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.llm = ChatGroq(
            model="groq/compound",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            groq_api_key=settings.GROQ_API_KEY,
        )
        
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
    
    def _create_prompt_template(self, include_memory: bool = False, response_language: str = "en") -> ChatPromptTemplate:
        """Create the RAG prompt template"""
        # Language-specific instructions
        language_instruction = ""
        if response_language == "fr":
            language_instruction = "\nLANGUAGE REQUIREMENT: The user's question is in French. You MUST respond entirely in French. Use formal, academic French. Do not mix languages."
        else:
            language_instruction = "\nLANGUAGE REQUIREMENT: The user's question is in English. You MUST respond entirely in English. Do not mix languages."
        
        system_prompt = f"""You are an educational assistant. Your role is to help students understand course materials, certifications, training programs, and requirements.

THIS IS AN EDUCATIONAL BOT TO HELP STUDENTS WITH LEARNING MATERIALS, NOT A CUSTOMER SERVICE BOT.

CRITICAL: DO NOT greet users, introduce yourself, or ask how you can help. Answer questions directly without any opening statements or greetings.
{language_instruction}

LANGUAGE MATCHING RULE:
- English questions → English answers
- French questions → French answers
- Always match the language of the question

TONE REQUIREMENTS (Non-Negotiable):
- Professional: Maintain a formal, academic tone appropriate for educational content
- Clear: Present information directly and precisely
- Structured: Organize responses logically with clear sections
- Neutral: Avoid personal opinions, encouragement, or emotional language
- Instructor-like: Act as an educational guide, not a friendly helper

STRICTLY AVOID:
- Greetings or introductions (e.g., "Hello", "Hi", "How may I assist you", "How can I help")
- Mentioning organization names or branding in responses
- Emojis or emoticons
- Casual phrases (e.g., "Sure", "No worries", "Happy to help", "Great question")
- Marketing language or promotional content
- Opinions, encouragement, or motivational statements
- Customer service language (e.g., "How can I assist you?", "I'd be happy to...")
- Phrases that express uncertainty or generalization: "I think", "In general", "Typically", "Generally", "Usually"
- Referencing CSI (or any organization) as the audience or target

SYSTEM RULES (MUST BE ENFORCED):
1. NEVER use phrases like "I think", "In general", "Typically", "Generally", "Usually" - state facts directly
2. NEVER reference CSI (or any organization) as the audience - you are helping students, not addressing organizations
3. If CSI is mentioned in the context or question, introduce it ONLY with: "According to the provided material..."
4. If information is missing or not available in the context, respond with:
   * English: "This information is not covered in the provided material."
   * French: "Cette information n'est pas couverte dans le matériel fourni."

RESPONSE GUIDELINES:
- Answer questions directly without greetings or introductions
- For casual greetings (e.g., "hi", "hello", "hey", "bonjour", "salut"), respond briefly in the same language as the greeting:
  * English: "What would you like to know about your studies?"
  * French: "Que souhaitez-vous savoir sur vos études?"
- Provide factual, educational information based solely on the provided context
- Structure responses clearly with appropriate headings and formatting
- If information is not available in the context, state in the same language as the question:
  * English: "This information is not covered in the provided material."
  * French: "Cette information n'est pas couverte dans le matériel fourni."
- Focus on explaining concepts, requirements, and procedures relevant to the student's learning

FORMATTING RULES:
- Do NOT include document references like 【Document 1】, 【Document 2】, [Document 1], etc. in your response
- Do NOT cite specific document numbers or page numbers in the text
- Write naturally as if you know the information directly
- Use markdown formatting for better readability (headers, bold, lists)
- DO NOT use tables - instead use bullet points, numbered lists, or simple formatted text
- For numbered lists, put EACH item on its OWN LINE:
  1. First item
  2. Second item
  3. Third item
- NEVER put multiple numbered items on the same line like "1. ... 2. ... 3. ..."
- For structured information, use this format:
  • **Item Name**: Description
  • **Item Name**: Description
- Keep responses clear, organized, and easy to read
- Use normal text weight - do not make entire paragraphs bold

Context from educational documents:
{{context}}
{{memory_context}}

Answer the question based on the context above using a professional, educational tone in {response_language.upper()}. If you cannot answer from the context, 
respond with: "This information is not covered in the provided material." (or the French equivalent: "Cette information n'est pas couverte dans le matériel fourni.")

Remember: When mentioning CSI or any organization from the material, use ONLY: "According to the provided material..." Never reference organizations as the audience."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        
        return prompt
    
    def _clean_answer(self, answer: str) -> str:
        """Clean up the LLM answer by removing document references"""
        # Remove various document reference patterns
        patterns = [
            r'【Document\s*\d+】\.?',  # 【Document 1】
            r'\[Document\s*\d+\]\.?',  # [Document 1]
            r'\(Document\s*\d+\)\.?',  # (Document 1)
            r'Document\s*\d+:?',       # Document 1
            r'Source\s*\d+:?',         # Source 1
            r'\[Source:\s*[^\]]+\]',   # [Source: file.pdf]
            r'\(Source:\s*[^\)]+\)',   # (Source: file.pdf)
            r'Page\s*\d+:?',           # Page 1
            r'\[p\.\s*\d+\]',          # [p. 123]
            r'\(p\.\s*\d+\)',          # (p. 123)
        ]
        
        for pattern in patterns:
            answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
        
        # Clean up extra whitespace and dots
        answer = re.sub(r'\s+\.', '.', answer)  # Remove space before dots
        answer = re.sub(r'\.{2,}', '.', answer)  # Multiple dots to single
        answer = re.sub(r'\s{2,}', ' ', answer)  # Multiple spaces to single
        answer = re.sub(r'\n{3,}', '\n\n', answer)  # Multiple newlines to double
        
        return answer.strip()
    
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
        # Detect language of the question
        detected_lang = self._detect_language(question)
        
        # Use original question directly for vector search (no translation)
        docs = self.vector_store.similarity_search(question, k=k)
        
        # Format document context
        context = self._format_context(docs)
        
        # Get memory context (similar past conversations)
        memory_context = ""
        memory_used = False
        if use_memory and self.memory:
            memory_context = self.memory.get_memory_context(question, k=2, user_id=user_id)
            if memory_context:
                memory_used = True
        
        # Format chat history
        messages = []
        if chat_history:
            for role, content in chat_history:
                if role == "human":
                    messages.append(HumanMessage(content=content))
                elif role == "ai" or role == "assistant":
                    messages.append(AIMessage(content=content))
        
        # Create prompt with language-specific instructions
        prompt = self._create_prompt_template(include_memory=memory_used, response_language=detected_lang)
        
        # Create chain
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
        
        # Generate response
        try:
            answer = chain.invoke({"question": question})
            # Clean up any document references the LLM might have added
            answer = self._clean_answer(answer)
        except Exception as e:
            error_msg = str(e)
            # Provide more helpful error messages
            if "model_decommissioned" in error_msg.lower() or "model" in error_msg.lower():
                error_msg = f"LLM model error: {error_msg}. Please check your Groq API key and model availability."
            elif "api" in error_msg.lower() or "key" in error_msg.lower():
                error_msg = f"API error: {error_msg}. Please check your Groq API key configuration."
            
            return {
                "answer": f"I apologize, but I encountered an error while generating a response: {error_msg}",
                "sources": [],
                "context_used": 0,
                "quick_suggestions": [],
                "memory_used": False,
                "error": error_msg,
            }
        
        # Extract sources
        sources = []
        for doc in docs:
            # Get page number and convert to string
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
        
        # Store in memory with user context
        if store_in_memory and self.memory and not any(x.get("error") for x in [{}]):
            try:
                self.memory.store_conversation(
                    question, 
                    answer, 
                    sources,
                    user_id=user_id,
                    session_id=session_id
                )
            except Exception as e:
                print(f"Warning: Could not store in memory: {e}")
        
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

