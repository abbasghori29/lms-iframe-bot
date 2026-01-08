"""
Main API router
"""
from fastapi import APIRouter

from app.api.v1.endpoints import chat, widget, speech

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(widget.router, prefix="/widget", tags=["widget"])
api_router.include_router(speech.router, prefix="/speech", tags=["speech"])

