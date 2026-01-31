"""
Speech-to-Text API endpoint using OpenAI gpt-4o-transcribe
"""
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional

router = APIRouter()

# Lazy load speech service
_speech_service = None


def get_speech():
    """Lazy load speech service"""
    global _speech_service
    if _speech_service is None:
        from app.services.speech import get_speech_service
        print("\n" + "=" * 60)
        print("🎤 LOADING SPEECH-TO-TEXT SERVICE (First Use)")
        print("=" * 60)
        print("Model: OpenAI gpt-4o-transcribe")
        print("")
        _speech_service = get_speech_service()
        print("")
        print("=" * 60)
        print("✓ Speech-to-Text service ready!")
        print("=" * 60 + "\n")
    return _speech_service


@router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
):
    """
    Transcribe audio file to text using OpenAI gpt-4o-transcribe.
    
    - **audio**: Audio file (supports webm, wav, mp3, m4a, etc.)
    - **language**: Optional language code (e.g., 'en', 'fr'). Auto-detects if not provided.
    
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
                detail=f"Speech service not available: {str(e)}"
            )
        
        # Get file extension from filename
        filename = audio.filename or "recording.webm"
        extension = filename.rsplit('.', 1)[-1] if '.' in filename else "webm"
        
        # Transcribe
        result = speech_service.transcribe(audio_data, language=language, file_extension=extension)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Transcription failed: {result.get('error', 'Unknown error')}"
            )
        
        return {
            "success": True,
            "text": result["text"],
            "language": result.get("language"),
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
            "model": "gpt-4o-transcribe",
            "initialized": speech_service.client is not None,
        }
    except Exception as e:
        return {
            "status": "unavailable",
            "error": str(e),
        }
