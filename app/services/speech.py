"""
Speech-to-Text Service using Faster Whisper
"""
import os
import tempfile
from typing import Optional
from pathlib import Path

from faster_whisper import WhisperModel

from app.core.config import settings


class SpeechToTextService:
    """
    Speech-to-Text service using Faster Whisper for transcription.
    Uses the 'base' model by default for balance of speed and accuracy.
    """
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize the Whisper model.
        
        Args:
            model_size: Model size - 'tiny', 'base', 'small', 'medium', 'large-v3'
                       Smaller = faster, Larger = more accurate
        """
        self.model_size = model_size
        self.model: Optional[WhisperModel] = None
        self._initialize()
    
    def _initialize(self):
        """Load the Whisper model"""
        try:
            print(f"Loading Whisper model: {self.model_size}...")
            self.model = WhisperModel(
                self.model_size,
                device="cpu",  # Use CPU for broader compatibility
                compute_type="int8",  # Use int8 for faster inference
            )
            print(f"✓ Whisper model '{self.model_size}' loaded successfully")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            raise
    
    def transcribe(self, audio_data: bytes, language: str = "en") -> dict:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Raw audio bytes (supports webm, wav, mp3, etc.)
            language: Language code (e.g., 'en', 'fr', 'es')
        
        Returns:
            Dictionary with transcription result
        """
        if not self.model:
            raise RuntimeError("Whisper model not initialized")
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        try:
            # Transcribe
            segments, info = self.model.transcribe(
                temp_path,
                language=language,
                beam_size=5,
                vad_filter=True,  # Voice Activity Detection for better results
            )
            
            # Combine all segments
            full_text = ""
            segment_list = []
            
            for segment in segments:
                full_text += segment.text + " "
                segment_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                })
            
            return {
                "success": True,
                "text": full_text.strip(),
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "segments": segment_list,
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
    
    def transcribe_file(self, file_path: str, language: str = "en") -> dict:
        """
        Transcribe an audio file.
        
        Args:
            file_path: Path to audio file
            language: Language code
        
        Returns:
            Dictionary with transcription result
        """
        with open(file_path, "rb") as f:
            audio_data = f.read()
        return self.transcribe(audio_data, language)


# Singleton instance
_speech_service: Optional[SpeechToTextService] = None


def get_speech_service(model_size: str = "base") -> SpeechToTextService:
    """Get or create speech service instance"""
    global _speech_service
    if _speech_service is None:
        _speech_service = SpeechToTextService(model_size=model_size)
    return _speech_service


def init_speech_service():
    """
    Initialize speech service on startup.
    This pre-downloads and loads the model so it's ready when users need it.
    Called from FastAPI lifespan startup.
    """
    from app.core.config import settings
    print("\n" + "=" * 60)
    print("🎤 INITIALIZING SPEECH-TO-TEXT SERVICE")
    print("=" * 60)
    print(f"Model: {settings.WHISPER_MODEL}")
    print("Note: First time may download the model (this can take a few minutes)")
    print("")
    
    try:
        service = get_speech_service(model_size=settings.WHISPER_MODEL)
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

