"""
Speech-to-Text API endpoint
"""
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional

router = APIRouter()

# Lazy load speech service to avoid loading model on startup
_speech_service = None


def get_speech():
    """Lazy load speech service"""
    global _speech_service
    if _speech_service is None:
        from app.services.speech import get_speech_service
        from app.core.config import settings
        print("\n" + "=" * 60)
        print("🎤 LOADING SPEECH-TO-TEXT SERVICE (First Use)")
        print("=" * 60)
        print(f"Model: {settings.WHISPER_MODEL}")
        print("Note: First time may download the model (this can take a few minutes)")
        print("")
        _speech_service = get_speech_service(model_size=settings.WHISPER_MODEL)
        print("")
        print("=" * 60)
        print("✓ Speech-to-Text service ready!")
        print("=" * 60 + "\n")
    return _speech_service


@router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(default="en"),
):
    """
    Transcribe audio file to text.
    
    - **audio**: Audio file (supports webm, wav, mp3, m4a, etc.)
    - **language**: Language code (default: 'en')
    
    Returns transcribed text.
    """
    try:
        # Read audio data
        audio_data = await audio.read()
        
        if len(audio_data) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty audio file"
            )
        
        # Get speech service
        try:
            speech_service = get_speech()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Speech service not available: {str(e)}. Model may still be loading."
            )
        
        # Transcribe
        result = speech_service.transcribe(audio_data, language=language)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Transcription failed: {result.get('error', 'Unknown error')}"
            )
        
        return {
            "success": True,
            "text": result["text"],
            "language": result.get("language"),
            "duration": result.get("duration"),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing audio: {str(e)}"
        )


@router.get("/health")
async def speech_health():
    """Check if speech service is available"""
    try:
        speech_service = get_speech()
        return {
            "status": "healthy",
            "model": speech_service.model_size,
            "initialized": speech_service.model is not None,
        }
    except Exception as e:
        return {
            "status": "unavailable",
            "error": str(e),
        }

