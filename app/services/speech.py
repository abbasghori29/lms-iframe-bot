"""
Speech-to-Text Service using OpenAI gpt-4o-transcribe
"""
import os
import tempfile
from typing import Optional

from openai import OpenAI

from app.core.config import settings


class SpeechToTextService:
    """
    Speech-to-Text service using OpenAI's gpt-4o-transcribe model.
    Fast, accurate, and supports multiple languages.
    """
    
    def __init__(self):
        """Initialize the OpenAI client."""
        self.client: Optional[OpenAI] = None
        self._initialize()
    
    def _initialize(self):
        """Initialize OpenAI client"""
        try:
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not configured")
            
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            print("✓ OpenAI Speech-to-Text service initialized (gpt-4o-transcribe)")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            raise
    
    def transcribe(self, audio_data: bytes, language: Optional[str] = None, file_extension: str = "webm") -> dict:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Raw audio bytes (supports webm, wav, mp3, m4a, etc.)
            language: Optional language code (e.g., 'en', 'fr'). If None, auto-detects.
            file_extension: File extension for the audio data (e.g., 'webm', 'mp4', 'ogg')
        
        Returns:
            Dictionary with transcription result
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        # Save audio to temporary file with correct extension
        with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        try:
            # Open the audio file
            with open(temp_path, "rb") as audio_file:
                # Transcribe using OpenAI
                transcription = self.client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=audio_file,
                    language=language,  # None = auto-detect
                )
            
            return {
                "success": True,
                "text": transcription.text,
                "language": language or "auto",
            }
            
        except Exception as e:
            return {
                "success": False,
                "text": "",
                "error": str(e),
            }
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def transcribe_file(self, file_path: str, language: Optional[str] = None) -> dict:
        """
        Transcribe an audio file.
        
        Args:
            file_path: Path to audio file
            language: Optional language code
        
        Returns:
            Dictionary with transcription result
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            with open(file_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=audio_file,
                    language=language,
                )
            
            return {
                "success": True,
                "text": transcription.text,
                "language": language or "auto",
            }
            
        except Exception as e:
            return {
                "success": False,
                "text": "",
                "error": str(e),
            }


# Singleton instance
_speech_service: Optional[SpeechToTextService] = None


def get_speech_service() -> SpeechToTextService:
    """Get or create speech service instance"""
    global _speech_service
    if _speech_service is None:
        _speech_service = SpeechToTextService()
    return _speech_service


def init_speech_service():
    """
    Initialize speech service on startup.
    Called from FastAPI lifespan startup.
    """
    print("\n" + "=" * 60)
    print("🎤 INITIALIZING SPEECH-TO-TEXT SERVICE")
    print("=" * 60)
    print("Model: OpenAI gpt-4o-transcribe")
    print("")
    
    try:
        service = get_speech_service()
        print("")
        print("=" * 60)
        print("✓ Speech-to-Text service ready!")
        print("=" * 60 + "\n")
        return service
    except Exception as e:
        print(f"\n❌ Failed to initialize speech service: {e}")
        print("Speech-to-text will be unavailable")
        print("=" * 60 + "\n")
        return None
