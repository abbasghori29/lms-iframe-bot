"""
Chatbot service with RAG (Retrieval Augmented Generation)
Includes memory for past Q&A
"""
import os
import sys
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from uuid import uuid4

# Add project root to path to import train_vector_db
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings
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
        # OpenAI-only LLM used for structured meta tasks (JSON outputs).
        # We intentionally do not use the Groq fallback for these calls.
        self.meta_llm = None
        self.vector_store = None
        self.embeddings = None
        self.memory: Optional[MemoryService] = None
        self._debug_buffer: List[str] = []
        # Small in-memory caches to avoid repeated LLM meta-calls during a session
        self._definition_cache: Dict[str, bool] = {}
        self._retrieval_hint_cache: Dict[str, List[str]] = {}
        self._subquestion_cache: Dict[str, List[str]] = {}
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
                print("OK: OpenAI LLM initialized as primary")
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
                print("OK: Groq LLM initialized as fallback")
            except Exception as e:
                print(f"Warning: Could not initialize Groq LLM: {e}")
        
        # Set up LLM with fallback chain
        if primary_llm and fallback_llm:
            self.llm = primary_llm.with_fallbacks([fallback_llm])
            print("OK: LLM configured with OpenAI primary + Groq fallback")
        elif primary_llm:
            self.llm = primary_llm
            print("OK: Using OpenAI LLM only (no fallback)")
        elif fallback_llm:
            self.llm = fallback_llm
            print("OK: Using Groq LLM only (OpenAI not configured)")
        else:
            raise ValueError("No LLM API key configured. Set OPENAI_API_KEY or GROQ_API_KEY in .env")

        # OpenAI-only meta LLM for strict JSON outputs (classification/extraction/hints).
        # This avoids brittle json.loads failures when the fallback LLM returns non-JSON.
        if settings.OPENAI_API_KEY:
            try:
                self.meta_llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    max_tokens=256,
                    timeout=None,
                    max_retries=2,
                    api_key=settings.OPENAI_API_KEY,
                    model_kwargs={"response_format": {"type": "json_object"}},
                )
            except Exception as e:
                print(f"Warning: Could not initialize meta LLM: {e}")
                self.meta_llm = None

        # Initialize OpenAI embeddings (much better semantic search quality)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=settings.OPENAI_API_KEY,
        )
        print("OK: Using OpenAI embeddings (text-embedding-3-large)")
        
        # Load vector store
        self._load_vector_store()
        
        # Initialize memory service (Pinecone with OpenAI embeddings)
        try:
            self.memory = get_memory_service(embeddings=self.embeddings)
            print(f"OK: Memory service initialized ({MEMORY_TYPE})")
        except Exception as e:
            print(f"Warning: Could not initialize memory service ({MEMORY_TYPE}): {e}")
            self.memory = None
    
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
            print(f"OK: Loaded vector store with {len(self.vector_store.index_to_docstore_id)} documents")
        except Exception as e:
            raise Exception(f"Error loading vector store: {e}")
    
    def _create_prompt_template(self, include_memory: bool = False) -> ChatPromptTemplate:
        """Create the RAG prompt template"""
        system_prompt = """You are a helpful, friendly tutor for CAFS (Canadian Association of Financial Services).
You must answer like a human tutor, but you can ONLY use the information in the Context below.
Do NOT use outside knowledge.
If the question is a follow-up, use any related info in the Context even if the wording differs.

If the Context contains related information, you may apply it to the user's question.
Do not refuse unless the Context is clearly unrelated to the question.

Language:
- Respond in the SAME language as the user (English or French).

Style:
- Be clear, conversational, and explanatory.
- If the question is broad, summarize the relevant parts from the Context.
- If the question has multiple parts, address each part briefly.

If context is insufficient, use this refusal:
- English: "I'm sorry, but I don't have information about this topic in my knowledge base. I can only provide answers based on the available CAFS documentation."
- French: "Je suis desole, mais je n'ai pas d'informations sur ce sujet dans ma base de connaissances. Je ne peux fournir que des reponses basees sur la documentation ACFS disponible."

Format: ONLY use HTML tags: <h3>, <p>, <ul>, <ol>, <li>, <strong>.

Context:
{context}
{memory_context}
"""
        
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
    
    def _ensure_html_wrapped(self, text: str) -> str:
        """Ensure the answer is fully wrapped in HTML tags"""
        stripped = text.strip()
        if not stripped:
            return "<p></p>"
        # If it doesn't contain any HTML tags, wrap in <p>
        if not re.search(r'<[^>]+>', stripped):
            return f"<p>{stripped}</p>"
        return text

    def _clean_answer(self, answer: str) -> str:
        """Clean up the LLM answer - normalize to HTML and remove junk"""
        
        # Step 1: Remove document references
        doc_patterns = [
            r'\[Document\s*\d+\]\.?',
            r'\(Document\s*\d+\)\.?',
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

        # Step 5: Ensure HTML wrapper exists
        answer = self._ensure_html_wrapped(answer)
        
        return answer

    def _expand_query(self, question: str) -> str:
        """Query expansion hook.

        We keep this intentionally minimal to avoid scenario-specific hardcoding.
        Retrieval recall improvements should come from the LLM-based retrieval hints
        (cached) and follow-up query stitching.
        """
        return question

    def _is_followup_question(self, question: str) -> bool:
        """Heuristic to detect short/vague follow-up questions"""
        q = question.strip().lower()
        # If the question contains a clear topic anchor, don't treat it as a follow-up
        topic_anchors = [
            "mutual fund",
            "bond fund",
            "equity",
            "negative interest rate",
            "rrsp",
            "fonds commun",
            "obligation",
            "taux d'interet negatif",
        ]
        if any(t in q for t in topic_anchors):
            return False
        if len(q.split()) <= 6:
            return True
        followup_starts = (
            "what about", "and what", "and then", "what would", "what happens",
            "what would happen", "what about my", "what about the", "and",
            "so", "then", "that", "this", "it"
        )
        return q.startswith(followup_starts)

    def _build_retrieval_query(self, question: str, chat_history: Optional[List]) -> str:
        """Build a retrieval query using follow-up context when helpful"""
        if not chat_history or not self._is_followup_question(question):
            return question

        last_user = None
        last_ai = None
        for role, content in reversed(chat_history):
            if last_ai is None and (role == "ai" or role == "assistant"):
                last_ai = content
            if last_user is None and role == "human":
                # Skip if the history includes the current user message
                if content.strip().lower() == question.strip().lower():
                    continue
                last_user = content
            if last_user and last_ai is not None:
                break

        if last_user:
            # Always include the last user question for short follow-ups to preserve topic.
            if last_ai:
                # Include a short excerpt of the last assistant answer to reinforce topic anchoring.
                ai_excerpt = re.sub(r"<[^>]+>", " ", last_ai)
                ai_excerpt = re.sub(r"\s+", " ", ai_excerpt).strip()[:200]
                return f"{question} | {last_user} | {ai_excerpt}"
            return f"{question} | {last_user}"

        return question
    
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

    def _llm_yes_no(self, prompt: str) -> Optional[bool]:
        """Ask the LLM a YES/NO question. Returns True/False or None on failure."""
        try:
            if not self.llm:
                return None
            resp = self.llm.invoke(prompt)
            txt = resp.content if hasattr(resp, "content") else str(resp)
            txt = (txt or "").strip().upper()
            if txt.startswith("YES"):
                return True
            if txt.startswith("NO"):
                return False
            return None
        except Exception as e:
            self._debug_log("llm_yes_no_error", str(e))
            return None

    def _meta_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call the OpenAI-only meta LLM in strict JSON mode and parse the result."""
        try:
            if not self.meta_llm:
                return None
            resp = self.meta_llm.invoke(prompt)
            txt = resp.content if hasattr(resp, "content") else str(resp)
            txt = (txt or "").strip()
            if not txt:
                return None
            try:
                data = json.loads(txt)
                return data if isinstance(data, dict) else None
            except Exception:
                # Last-resort: extract first {...} block.
                m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
                if not m:
                    return None
                data = json.loads(m.group(0))
                return data if isinstance(data, dict) else None
        except Exception as e:
            self._debug_log("meta_json_error", str(e))
            return None

    def _groundness_gate_llm(self, question: str, answer: str, context: str) -> Optional[bool]:
        """Model-based groundedness gate: verify answer claims are supported by context."""
        try:
            if not self.meta_llm or not question or not answer or not context:
                return None
            prompt = (
                "You are a strict verifier.\n"
                "Return ONLY a JSON object.\n"
                "Decide if every major claim in the Answer is directly supported by the Context.\n"
                "If the Answer adds details not in the Context, mark grounded=false.\n\n"
                f"Question:\n{question}\n\n"
                f"Answer:\n{answer}\n\n"
                f"Context:\n{context}\n\n"
                "JSON schema:\n"
                "{\"grounded\": true|false}"
            )
            data = self._meta_json(prompt)
            if not data or "grounded" not in data:
                return None
            return bool(data.get("grounded"))
        except Exception as e:
            self._debug_log("groundness_gate_error", str(e))
            return None

    def _is_definition_question_llm(self, question: str) -> Optional[bool]:
        """LLM-based definition classifier (strict JSON). Cached.

        This detects *definition-style* questions (short meaning/definition).
        Broad explanation prompts and multi-part questions should return False.
        """
        key = re.sub(r"\s+", " ", question.strip().lower())
        if key in self._definition_cache:
            return self._definition_cache[key]

        prompt = (
            "You are a classifier.\n"
            "Return a JSON object only.\n"
            "Task: Decide whether the user is asking for a SHORT definition/meaning of a term or concept.\n"
            "Return true only for definition-style questions like:\n"
            "- \"What is X?\" / \"Define X\" / \"What does X mean?\" / \"Qu'est-ce que X?\"\n"
            "Return false for:\n"
            "- broad explanation prompts (\"tell me about...\", \"explain...\"),\n"
            "- multi-part/comparison questions (contain \"and\", \"compare\", \"vs\"),\n"
            "- procedural/how-to questions (\"how do I...\"),\n"
            "- questions asking for effects/impacts (\"how does... affect...\").\n"
            f"User question: {question}\n"
            "JSON schema:\n"
            "{\"is_definition\": true|false}"
        )
        data = self._meta_json(prompt)
        if not data or "is_definition" not in data:
            return None
        val = bool(data.get("is_definition"))
        self._definition_cache[key] = val
        return val

    def _retrieval_hints_llm(self, question: str) -> List[str]:
        """Ask the LLM for short retrieval hint phrases to improve search recall. Cached."""
        key = re.sub(r"\s+", " ", question.strip().lower())
        if key in self._retrieval_hint_cache:
            return self._retrieval_hint_cache[key]
        try:
            if not self.meta_llm:
                return []
            prompt = (
                "You generate search hints for a document retriever.\n"
                "Return ONLY a JSON object.\n"
                "Rules:\n"
                "- Provide 0 to 5 short hint phrases.\n"
                "- No sentences; no advice.\n"
                "- Include synonyms, Canadian/US variants, and closely related textbook phrases.\n"
                f"User question: {question}\n"
                "JSON schema:\n"
                "{\"hints\": [\"...\"]}"
            )
            data = self._meta_json(prompt) or {}
            raw = data.get("hints", [])
            hints = [s.strip() for s in raw if isinstance(s, str) and s.strip()]
            hints = hints[:5]
            self._retrieval_hint_cache[key] = hints
            return hints
        except Exception as e:
            self._debug_log("retrieval_hints_error", str(e))
            return []

    def _subquestions_llm(self, question: str) -> List[str]:
        """Decompose a complex question into a few focused sub-questions (strict JSON). Cached."""
        key = re.sub(r"\s+", " ", question.strip().lower())
        if key in self._subquestion_cache:
            return self._subquestion_cache[key]
        try:
            if not self.meta_llm:
                return []
            prompt = (
                "You decompose user questions into focused sub-questions to help retrieve evidence.\n"
                "Return ONLY a JSON object.\n"
                "Rules:\n"
                "- Provide 1 to 4 sub-questions.\n"
                "- Keep them short.\n"
                "- Preserve the user's intent and key terms.\n"
                "- No extra commentary.\n"
                f"User question: {question}\n"
                "JSON schema:\n"
                "{\"subquestions\": [\"...\"]}"
            )
            data = self._meta_json(prompt) or {}
            raw = data.get("subquestions", [])
            subqs = [s.strip() for s in raw if isinstance(s, str) and s.strip()]
            subqs = subqs[:4]
            self._subquestion_cache[key] = subqs
            return subqs
        except Exception as e:
            self._debug_log("subquestions_error", str(e))
            return []

    def _retrieval_hints_from_text_llm(self, text: str, label: str = "context") -> List[str]:
        """Generate retrieval hints from a short context snippet (strict JSON)."""
        try:
            if not self.meta_llm or not text:
                return []
            snippet = re.sub(r"\s+", " ", text).strip()[:500]
            prompt = (
                "You generate search hints for a document retriever.\n"
                "Return ONLY a JSON object.\n"
                "Rules:\n"
                "- Provide 0 to 5 short hint phrases.\n"
                "- No sentences; no advice.\n"
                "- Extract key phrases and synonyms implied by the snippet.\n"
                f"Snippet ({label}): {snippet}\n"
                "JSON schema:\n"
                "{\"hints\": [\"...\"]}"
            )
            data = self._meta_json(prompt) or {}
            raw = data.get("hints", [])
            hints = [s.strip() for s in raw if isinstance(s, str) and s.strip()]
            return hints[:5]
        except Exception as e:
            self._debug_log("retrieval_hints_from_text_error", str(e))
            return []
    
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

    def _is_no_information(self, answer: str) -> bool:
        """Detect the standard 'no information' refusal response"""
        if not answer:
            return False
        lower = answer.lower()
        return (
            "don't have information" in lower
            or "do not have information" in lower
            or "no information available" in lower
        )

    def _followup_fallback_from_context(self, question: str, context: str, chat_history: Optional[List]) -> Optional[str]:
        """If a follow-up refused, try to answer from a clearly relevant sentence in context."""
        if not chat_history or not self._is_followup_question(question):
            return None
        q = question.lower()
        if not ("account" in q or "chequing" in q or "checking" in q or "deposit" in q):
            return None
        # Find a sentence in context that mentions charging for deposits.
        ctx = re.sub(r"\s+", " ", context)
        sentences = re.split(r"(?<=[\.!?])\s+", ctx)
        for s in sentences:
            sl = s.lower()
            if ("charge" in sl or "charges" in sl) and ("deposit" in sl or "deposits" in sl):
                return f"<p>{s.strip()}</p>"
        return None

    def _replace_no_information(self, answer: str, question: str) -> str:
        """Replace terse refusal with a friendly, HTML-wrapped message"""
        if not answer:
            return answer
        # Normalize and strip HTML to catch refusal variants
        cleaned = re.sub(r"<[^>]+>", " ", answer).strip().lower()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if "no information available" in cleaned or "don't have information" in cleaned or "do not have information" in cleaned:
            return self._refusal_message(question)
        return answer

    def _debug_log(self, label: str, data: str) -> None:
        """Append debug info to retrieval_debug.log immediately"""
        try:
            log_path = Path(project_root) / "retrieval_debug.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[DEBUG] {label}: {data}\n")
            # Also print to console for immediate visibility
            print(f"[DEBUG] {label}: {data}", flush=True)
        except Exception:
            # Fallback to console if file write fails
            print(f"[DEBUG] {label}: {data} (file write failed)", flush=True)

    def _flush_debug_log(self, question: str) -> None:
        """Flush buffered debug info to retrieval_debug.log"""
        if not self._debug_buffer:
            return
        try:
            log_path = Path(project_root) / "retrieval_debug.log"
            with open(log_path, "a", encoding="utf-8") as f:
                for line in self._debug_buffer:
                    f.write(line + "\n")
        except Exception:
            pass
        finally:
            self._debug_buffer = []

    def _refusal_message(self, question: str) -> str:
        """Human-friendly refusal message in the user's language"""
        lang = self._detect_language(question)
        if lang == "fr":
            return (
                "<p>Desole, je ne peux repondre qu'a partir de la documentation CAFS "
                "dont je dispose. Je n'ai pas d'information sur ce sujet pour le moment. "
                "Si vous posez une question liee aux contenus CAFS, je serai ravi d'aider.</p>"
            )
        return (
            "<p>Sorry, I can only answer from the CAFS documentation I have. "
            "I don't have information on this topic right now. "
            "If you ask something tied to the CAFS materials, I'll be happy to help.</p>"
        )

    def _groundedness_gate(self, question: str, context: str) -> bool:
        """Use the LLM to decide if the context contains an answer"""
        try:
            if not self.llm or not context:
                return False
            gate_prompt = (
                "You are a strict verifier. Answer ONLY with YES or NO.\n"
                "Question:\n"
                f"{question}\n\n"
                "Context:\n"
                f"{context}\n\n"
                "Does the context contain enough information to answer the question?\n"
                "Respond with YES or NO only."
            )
            resp = self.llm.invoke(gate_prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            text = text.strip().upper()
            return text.startswith("YES")
        except Exception as e:
            print(f"Warning: groundedness gate failed: {e}")
            return False

    def _supporting_sentences_check(self, question: str, answer: str, context: str) -> bool:
        """Ask the LLM to verify each major claim is supported by context"""
        try:
            if not self.llm or not context or not answer:
                return False
            verify_prompt = (
                "You are a strict verifier. Answer ONLY with YES or NO.\n"
                "Question:\n"
                f"{question}\n\n"
                "Answer:\n"
                f"{answer}\n\n"
                "Context:\n"
                f"{context}\n\n"
                "Does every major claim in the Answer have direct support in the Context?\n"
                "Respond with YES or NO only."
            )
            resp = self.llm.invoke(verify_prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            text = text.strip().upper()
            return text.startswith("YES")
        except Exception as e:
            print(f"Warning: supporting-sentences check failed: {e}")
            return False

    def _extract_supporting_sentences(self, question: str, context: str) -> List[str]:
        """Extract exact supporting sentences from context (verbatim).

        Uses the OpenAI-only meta LLM in strict JSON mode to avoid brittle JSON parsing.
        """
        try:
            if not context or not self.meta_llm:
                return []
            extract_prompt = (
                "You are a strict extractor.\n"
                "Return ONLY a JSON object.\n"
                "Extract 0 to 8 exact sentences copied verbatim from the Context that directly answer the Question.\n"
                "Do not paraphrase. Do not add anything.\n"
                "If nothing directly answers it, return an empty list.\n\n"
                f"Question:\n{question}\n\n"
                f"Context:\n{context}\n\n"
                "JSON schema:\n"
                "{\"sentences\": [\"...\"]}"
            )
            data = self._meta_json(extract_prompt) or {}
            sentences = data.get("sentences", [])
            if not isinstance(sentences, list):
                return []

            # Keep only sentences that appear verbatim (whitespace-normalized) in context
            def norm(s: str) -> str:
                return re.sub(r"\s+", " ", (s or "").strip())

            norm_context = norm(context)
            verified = []
            for s in sentences:
                if not isinstance(s, str):
                    continue
                ns = norm(s)
                if ns and ns in norm_context:
                    verified.append(s.strip())
            return verified
        except Exception as e:
            print(f"Warning: supporting sentence extraction failed: {e}")
            return []

    def _dedupe_sentences(self, sentences: List[str]) -> List[str]:
        """De-duplicate near-identical sentences by aggressive normalization"""
        seen = set()
        result = []
        for s in sentences:
            if not s:
                continue
            norm = s.lower()
            norm = re.sub(r"[^\w\s]", " ", norm)
            norm = re.sub(r"\bthe\b|\ba\b|\ban\b", " ", norm)
            norm = re.sub(r"\s+", " ", norm).strip()
            if norm in seen:
                continue
            seen.add(norm)
            result.append(s.strip())
        return result

    def _first_sentence_only(self, text: str) -> str:
        """Return only the first sentence from a text block"""
        if not text:
            return text
        # Simple sentence split on period/question/exclamation
        parts = re.split(r"(?<=[\.!?])\s+", text.strip())
        return parts[0] if parts else text.strip()

    def _jaccard_dedupe(self, sentences: List[str], threshold: float = 0.4) -> List[str]:
        """De-duplicate sentences using token Jaccard similarity"""
        def tokens(s: str) -> set:
            s = s.lower()
            s = re.sub(r"[^\w\s]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return set(t for t in s.split(" ") if t)

        kept = []
        kept_tokens = []
        for s in sentences:
            t = tokens(s)
            duplicate = False
            for kt in kept_tokens:
                if not t or not kt:
                    continue
                inter = len(t & kt)
                union = len(t | kt)
                if union > 0 and (inter / union) >= threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(s.strip())
                kept_tokens.append(t)
        return kept

    def _extract_keywords(self, question: str) -> List[str]:
        """Extract simple keywords from the question for filtering"""
        q = question.lower()
        # Normalize common Unicode punctuation to ASCII using escapes (keeps source ASCII-safe)
        q = q.replace("\u2019", "'").replace("\u2018", "'").replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
        q = re.sub(r"[^\w\s-]", " ", q)
        tokens = [t for t in q.split() if len(t) >= 4]
        # Keep domain-relevant tokens even if common
        keep = {"bond", "bonds", "equity", "equities", "interest", "rate", "rates", "duration"}
        stop = {
            "what", "what's", "what", "which", "how", "when", "where", "who", "why",
            "tell", "about", "compare", "difference", "between", "would", "could",
            "does", "they", "them", "their", "this", "that", "with", "your",
            "qu", "que", "quoi", "comment", "pourquoi", "avec", "dans", "pour",
            "fonds", "commun", "placement", "mutual", "fund", "funds"
        }
        return [t for t in tokens if (t in keep) or (t not in stop)]

    def _required_terms(self, question: str) -> List[str]:
        """Return required domain terms inferred from the question"""
        q = question.lower()
        terms = []
        mapping = [
            ("bond", "bond"),
            ("bonds", "bond"),
            ("equity", "equity"),
            ("equities", "equity"),
            ("interest rate", "interest rate"),
            ("interest rates", "interest rate"),
            ("rates", "rate"),
        ]
        for key, term in mapping:
            if key in q and term not in terms:
                terms.append(term)
        return terms

    def _required_topics(self, question: str) -> List[str]:
        """Infer required topic coverage for complex questions (risk, interest rates, etc.)"""
        q = question.lower()
        req = []
        if "risk" in q or "volatil" in q:
            req.append("risk")
        if "interest rate" in q or "interest rates" in q or "taux" in q:
            req.append("interest_rate")
        if "equity" in q or "equities" in q:
            req.append("equity")
        if "bond" in q or "bonds" in q:
            req.append("bond")
        return req

    def _covers_required_topics(self, question: str, sentences: List[str], required_topics: List[str]) -> bool:
        """Check if extracted sentences cover required topics (optionally requiring directional effects)."""
        if not required_topics:
            return True
        q = question.lower()
        text = " ".join(sentences).lower()
        # Directional check for interest-rate effect questions (require explicit up/down statement).
        def has_directional_interest(sents: List[str]) -> bool:
            pat_up_down = re.compile(
                r"(interest rates?|yields?).{0,80}(rise|rising|increase|increases|higher|go up|moves? up).{0,120}(value|price|prices).{0,80}(fall|falls|decline|declines|decrease|decreases|drop|drops|lower)",
                re.IGNORECASE | re.DOTALL,
            )
            pat_down_up = re.compile(
                r"(interest rates?|yields?).{0,80}(fall|falls|decline|declines|decrease|decreases|lower|go down|moves? down).{0,120}(value|price|prices).{0,80}(rise|rises|increase|increases|higher|go up|moves? up)",
                re.IGNORECASE | re.DOTALL,
            )
            pat_fr_up_down = re.compile(
                r"(taux|rendements?).{0,80}(augment|hausse|monte).{0,120}(valeur|prix).{0,80}(baisse|diminu|recule)",
                re.IGNORECASE | re.DOTALL,
            )
            pat_fr_down_up = re.compile(
                r"(taux|rendements?).{0,80}(baisse|diminu|recule).{0,120}(valeur|prix).{0,80}(augment|hausse|monte)",
                re.IGNORECASE | re.DOTALL,
            )
            for s in sents:
                if pat_up_down.search(s) or pat_down_up.search(s) or pat_fr_up_down.search(s) or pat_fr_down_up.search(s):
                    return True
            return False
        directional_interest = has_directional_interest(sentences)
        checks = {
            "risk": ("risk" in text) or ("volatil" in text),
            "interest_rate": ("interest rate" in text) or ("interest rates" in text) or ("taux" in text) or ("duration" in text),
            "equity": ("equity" in text) or ("equities" in text) or ("actions" in text),
            "bond": ("bond" in text) or ("bonds" in text) or ("obligation" in text) or ("obligations" in text),
        }

        ok = all(checks.get(t, True) for t in required_topics)

        # If the question explicitly asks how interest rates affect something, require a directional statement.
        asks_effect = ("affect" in q or "impact" in q or "what happens" in q or "when interest rates" in q or "lorsque" in q)
        if asks_effect and "interest_rate" in required_topics:
            ok = ok and directional_interest

        return ok

    def _is_complex_question(self, question: str) -> bool:
        """Detect multi-part or comparison questions"""
        q = question.strip().lower()
        if " and " in q or "compare" in q or "difference" in q or "vs" in q:
            return True
        if "how do" in q and " affect " in q:
            return True
        return False

    def _context_has_procedural_steps(self, question: str, context: str) -> bool:
        """Ask the LLM if context explicitly contains procedural steps/instructions"""
        try:
            if not self.llm or not context:
                return False
            prompt = (
                "You are a strict verifier. Answer ONLY with YES or NO.\n"
                "Question:\n"
                f"{question}\n\n"
                "Context:\n"
                f"{context}\n\n"
                "Does the context explicitly describe steps or procedures to accomplish the question?\n"
                "Respond with YES or NO only."
            )
            resp = self.llm.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            text = text.strip().upper()
            return text.startswith("YES")
        except Exception as e:
            print(f"Warning: procedural-steps check failed: {e}")
            return False

    def _has_unverified_claim_patterns(self, answer: str, context: str) -> bool:
        """Detect risky claims (regulatory refs, timelines, amounts) not present in context"""
        if not answer or not context:
            return False
        a = answer.lower()
        c = context.lower()

        patterns = [
            "ni 81-102",
            "national instrument 81-102",
            "two business days",
            "2 business days",
            "minimum",
            "plan de retrait",
            "plans de retrait",
            "plans d'investissement",
            "plans d'investissement",
            "regular contribution",
            "pre-authorized contribution",
            "pac",
            "systematic withdrawal",
            "systematic withdrawal plan",
            "investor sentiment",
            "vicious cycle",
            "diversify",
            "diversification",
            "diversified",
            "diversified portfolio",
        ]

        # Amounts like $100 or 100 $ should appear in context
        import re
        amount_matches = re.findall(r"(?:\\$\\s*\\d+|\\d+\\s*\\$)", a)
        if amount_matches:
            if not any(m.lower() in c for m in amount_matches):
                return True

        for p in patterns:
            if p in a and p not in c:
                return True

        return False

    def _topic_mismatch(self, question: str, answer: str) -> bool:
        """Detect if answer drifts to an unrelated topic based on key nouns"""
        q = question.lower()
        a = answer.lower()
        topics = [
            ("mutual fund", "mutual fund"),
            ("fonds commun", "fonds commun"),
            ("bond fund", "bond"),
            ("negative interest rate", "negative interest"),
            ("chequing account", "account"),
            ("rrsp", "rrsp"),
        ]
        for q_key, a_key in topics:
            if q_key in q and a_key not in a:
                return True
        return False

    def _is_procedural_question(self, question: str) -> bool:
        """Detect procedural/how-to questions"""
        q = question.strip().lower()
        triggers = (
            "how do i", "how to", "steps", "process", "procedure",
            "what are the steps", "how can i", "how should i"
        )
        return any(q.startswith(t) for t in triggers)

    def _is_definition_question(self, question: str) -> bool:
        """Detect definition-style questions (EN/FR), with LLM fallback for edge cases."""
        q = question.strip().lower()
        # Normalize apostrophes and dashes (Unicode-safe via escapes)
        q = q.replace("\u2019", "'").replace("\u2018", "'").replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
        # Remove punctuation for robust matching
        q_norm = re.sub(r"[^a-z0-9'\s-]", " ", q)
        q_norm = re.sub(r"\s+", " ", q_norm).strip()

        triggers = (
            "what is", "what is a", "what is an", "define", "definition of",
            "qu'est-ce que", "qu'est-ce qu'un", "qu'est-ce qu'une", "qu est ce que",
            "c'est quoi", "c est quoi",
        )
        if any(q_norm.startswith(t) for t in triggers):
            return True

        # LLM fallback for cases like "what are X", "explain X", etc.
        maybe_definition = q_norm.startswith("what ") or q_norm.startswith("explain ") or q_norm.startswith("describe ")
        if maybe_definition:
            llm_val = self._is_definition_question_llm(question)
            if llm_val is not None:
                self._debug_log("definition_llm", "yes" if llm_val else "no")
                return llm_val
        return False

    def _regenerate_with_context(self, question: str, context: str, memory_context: str, chat_history: Optional[List]) -> str:
        """Regenerate an answer with an explicit instruction that context contains the answer"""
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful educational assistant for CAFS (Canadian Association of Financial Services).\n\n"
             "The Context below DOES contain the answer. You MUST answer using ONLY the Context.\n"
             "Do NOT refuse. Do NOT add outside knowledge. Match the user's language.\n"
             "Format using ONLY HTML tags: <h3>, <p>, <ul>, <ol>, <li>, <strong>.\n\n"
             "Context:\n{context}\n{memory_context}"
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        messages = []
        if chat_history:
            for role, content in chat_history:
                if role == "human":
                    messages.append(HumanMessage(content=content))
                elif role == "ai" or role == "assistant":
                    messages.append(AIMessage(content=content))
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
        answer = chain.invoke({"question": question})
        return self._clean_answer(answer)
    
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
        request_id = str(uuid4())

        # Log request header
        self._debug_log("request_id", request_id)
        self._debug_log("question", question)
        
        # Helper functions for parallel execution
        def do_vector_search():
            # Use scores for better selection on short/vague or follow-up queries
            base_query = self._build_retrieval_query(question, chat_history)
            query_expanded = self._expand_query(base_query)
            local_k = k
            if self._is_followup_question(question):
                local_k = min(max(k, 15), 30)
                self._debug_log("followup_k", str(local_k))

            # NOTE: Retrieval hints removed for speed. We rely on follow-up query stitching instead.

            docs_scores_original = self.vector_store.similarity_search_with_score(base_query, k=local_k)
            docs_scores_expanded = self.vector_store.similarity_search_with_score(query_expanded, k=local_k)

            # Choose the set with the better (lower) best score
            best_original = docs_scores_original[0][1] if docs_scores_original else float("inf")
            best_expanded = docs_scores_expanded[0][1] if docs_scores_expanded else float("inf")

            short_or_followup = self._is_followup_question(question)
            if best_expanded + 0.15 < best_original or short_or_followup:
                primary = docs_scores_expanded
                secondary = docs_scores_original
            else:
                primary = docs_scores_original
                secondary = docs_scores_expanded

            # If top score is weak, do a simple merge for better coverage
            weak_threshold = 1.1
            if primary and primary[0][1] > weak_threshold and secondary:
                merged = primary + secondary
            else:
                merged = primary

            # De-duplicate by content
            seen = set()
            docs = []
            debug_sources = []
            for doc, _score in merged:
                key = hash(doc.page_content)
                if key in seen:
                    continue
                seen.add(key)
                docs.append(doc)
                debug_sources.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page_number") or doc.metadata.get("page"),
                    "score": float(_score),
                })
                if len(docs) >= k:
                    break
            return docs, {
                "base_query": base_query,
                "expanded_query": query_expanded,
                "best_original": float(best_original),
                "best_expanded": float(best_expanded),
                "sources": debug_sources,
            }
        
        def do_memory_search():
            if use_memory and self.memory:
                return self.memory.get_memory_context(question, k=2, user_id=user_id)
            return ""
        
        # Step 1+2: Run vector search and memory search in PARALLEL
        step_start = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(do_vector_search)
            memory_future = executor.submit(do_memory_search)
            
            docs, search_debug = vector_future.result()
            memory_context = memory_future.result()
        
        parallel_time = (time.time() - step_start) * 1000
        print(f"[TIMING] [1+2] Vector + Memory search (PARALLEL): {parallel_time:.1f}ms", flush=True)
        
        memory_used = bool(memory_context)

        # Retrieval debug logging (to file)
        try:
            log_path = Path(project_root) / "retrieval_debug.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{'='*80}\n")
                f.write(f"{datetime.now().isoformat()} | Q: {question} | request_id: {request_id}\n")
                f.write(f"Base query: {search_debug.get('base_query')}\n")
                f.write(f"Expanded query: {search_debug.get('expanded_query')}\n")
                f.write(f"Best score (original): {search_debug.get('best_original')}\n")
                f.write(f"Best score (expanded): {search_debug.get('best_expanded')}\n")
                f.write("Top sources:\n")
                for s in search_debug.get("sources", []):
                    f.write(f"  - {s.get('source')} | page {s.get('page')} | score {s.get('score')}\n")
        except Exception as e:
            print(f"Warning: could not write retrieval log: {e}")

        # Flush any buffered debug lines
        self._flush_debug_log(question)
        
        # Step 3: Format context
        step_start = time.time()
        context = self._format_context(docs)
        print(f"[TIMING] [3] Format context: {(time.time() - step_start)*1000:.1f}ms", flush=True)

        # If this is a follow-up, append the last assistant reply to strengthen topic continuity.
        if chat_history and self._is_followup_question(question):
            last_ai = None
            for role, content in reversed(chat_history):
                if role == "ai" or role == "assistant":
                    last_ai = content
                    break
            if last_ai:
                ai_excerpt = re.sub(r"<[^>]+>", " ", last_ai)
                ai_excerpt = re.sub(r"\s+", " ", ai_excerpt).strip()
                if ai_excerpt:
                    context = context + "\n---\n" + ai_excerpt
                    self._debug_log("followup_context_added", "yes")

        # Simple confidence gate: if retrieval score is weak, refuse (keeps model within dataset).
        try:
            best_score = min(float(search_debug.get("best_original", 999)), float(search_debug.get("best_expanded", 999)))
        except Exception:
            best_score = 999
        if best_score > 1.1 and not memory_context:
            return {
                "answer": self._replace_no_information(self._refusal_message(question), question),
                "sources": [],
                "context_used": len(docs),
                "quick_suggestions": [],
                "memory_used": bool(memory_context),
                "user_id": user_id,
                "session_id": session_id,
            }

        # Step 4: Format chat history
        step_start = time.time()
        messages = []
        if chat_history:
            for role, content in chat_history:
                if role == "human":
                    messages.append(HumanMessage(content=content))
                elif role == "ai" or role == "assistant":
                    messages.append(AIMessage(content=content))
        print(f"[TIMING] [4] Format chat history: {(time.time() - step_start)*1000:.1f}ms", flush=True)
        
        # Step 5: Create prompt
        step_start = time.time()
        prompt = self._create_prompt_template(include_memory=memory_used)
        print(f"[TIMING] [5] Create prompt: {(time.time() - step_start)*1000:.1f}ms", flush=True)
        
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
        print(f"[TIMING] [6] Create chain: {(time.time() - step_start)*1000:.1f}ms", flush=True)
        
        # Step 7: LLM inference (THE BIG ONE)
        step_start = time.time()
        try:
            answer = chain.invoke({"question": question})
            llm_time = (time.time() - step_start)*1000
            print(f"[TIMING] [7] LLM inference: {llm_time:.1f}ms {'SLOW!' if llm_time > 3000 else ''}", flush=True)
            # Step 8: Clean answer
            step_start = time.time()
            answer = self._clean_answer(answer)
            print(f"[TIMING] [8] Clean answer: {(time.time() - step_start)*1000:.1f}ms", flush=True)
            # Replace terse refusal with a friendlier message
            before = answer
            answer = self._replace_no_information(answer, question)
            if answer != before:
                self._debug_log("refusal_replaced", "true")
            else:
                if self._is_no_information(answer):
                    self._debug_log("refusal_replaced", "false")

            # If we still have a refusal on a follow-up, try a minimal context-based fallback.
            if self._is_no_information(answer):
                fallback = self._followup_fallback_from_context(question, context, chat_history)
                if fallback:
                    self._debug_log("followup_fallback_used", "true")
                    answer = fallback
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] LLM error after {(time.time() - step_start)*1000:.1f}ms: {error_msg}", flush=True)
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
        print(f"[TIMING] [9] Extract sources: {(time.time() - step_start)*1000:.1f}ms", flush=True)
        
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
                print(f"[TIMING] [BG] Memory stored successfully", flush=True)
            except Exception as e:
                print(f"Warning: Could not store in memory: {e}")
        
        if store_in_memory and self.memory and not any(x.get("error") for x in [{}]):
            import threading
            thread = threading.Thread(target=store_in_background, daemon=True)
            thread.start()
            print(f"[TIMING] [10] Store in memory: BACKGROUND (non-blocking)", flush=True)
        
        # Total time
        total_time = (time.time() - total_start)*1000
        print(f"[TIMING] ===============================", flush=True)
        print(f"[TIMING] TOTAL TIME: {total_time:.1f}ms ({total_time/1000:.2f}s)", flush=True)
        print(f"[TIMING] ===============================", flush=True)
        
        return {
            "answer": self._replace_no_information(answer, question),
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
