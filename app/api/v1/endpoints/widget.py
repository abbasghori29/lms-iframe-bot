"""
Chatbot Widget endpoints - serves the iframe-based chatbot UI
"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.core.config import settings

router = APIRouter()

# Setup templates
templates_path = Path(__file__).parent.parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_path))


def get_base_url(request: Request) -> tuple:
    """Get base URL from request"""
    host = request.headers.get("host", f"localhost:{settings.PORT}")
    scheme = request.headers.get("x-forwarded-proto", "http")
    return scheme, host


@router.get("/widget", response_class=HTMLResponse)
async def get_widget(request: Request):
    """
    Serve the chatbot widget HTML (to be loaded in iframe)
    """
    scheme, host = get_base_url(request)
    api_url = f"{scheme}://{host}/api/v1/chat/chat"
    speech_api_url = f"{scheme}://{host}/api/v1/speech/transcribe"
    
    return templates.TemplateResponse(
        "chatbot/widget.html",
        {
            "request": request,
            "api_url": api_url,
            "speech_api_url": speech_api_url,
        }
    )


@router.get("/embed", response_class=HTMLResponse)
async def get_embed(request: Request):
    """
    Serve the embed HTML with toggle button and iframe container
    """
    scheme, host = get_base_url(request)
    widget_url = f"{scheme}://{host}/api/v1/widget/widget"
    
    return templates.TemplateResponse(
        "chatbot/embed.html",
        {
            "request": request,
            "widget_url": widget_url,
        }
    )


@router.get("/embed.js", response_class=HTMLResponse)
async def get_embed_script(request: Request):
    """
    Serve a JavaScript snippet for easy embedding on any website.
    Includes responsive sizing for all screen sizes.
    """
    scheme, host = get_base_url(request)
    embed_url = f"{scheme}://{host}/api/v1/widget/embed"
    
    script = f"""
(function() {{
    // CAFS Chatbot Embed Script - Responsive Version
    if (document.getElementById('cafs-chatbot-container')) return;
    
    var container = document.createElement('div');
    container.id = 'cafs-chatbot-container';
    container.style.cssText = 'position:fixed;bottom:0;right:0;z-index:999999;background:transparent;';
    
    var iframe = document.createElement('iframe');
    iframe.id = 'cafs-chatbot-iframe';
    iframe.src = '{embed_url}';
    iframe.style.cssText = 'border:none;background:transparent;';
    iframe.allow = 'microphone';
    iframe.title = 'CAFS Chatbot';
    
    container.appendChild(iframe);
    document.body.appendChild(container);
}})();
"""
    
    return HTMLResponse(
        content=script,
        media_type="application/javascript"
    )


@router.get("/demo", response_class=HTMLResponse)
async def get_demo(request: Request):
    """
    Serve a demo page showcasing the chatbot widget
    """
    scheme, host = get_base_url(request)
    
    return templates.TemplateResponse(
        "chatbot/demo.html",
        {
            "request": request,
            "embed_url": f"{scheme}://{host}/api/v1/widget/embed",
            "embed_js_url": f"{scheme}://{host}/api/v1/widget/embed.js",
        }
    )

