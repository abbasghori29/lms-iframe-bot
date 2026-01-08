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
    
    // Calculate responsive dimensions (narrower, taller)
    function getResponsiveDimensions() {{
        var vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
        var vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);
        
        var width, height;
        
        if (vw <= 360) {{
            width = Math.min(300, vw - 20);
            height = Math.min(500, vh - 20);
        }} else if (vw <= 480) {{
            width = Math.min(320, vw - 24);
            height = Math.min(540, vh - 24);
        }} else if (vw <= 768) {{
            width = Math.min(340, vw - 32);
            height = Math.min(580, vh - 40);
        }} else if (vw <= 1024) {{
            width = 360;
            height = Math.min(600, vh - 40);
        }} else {{
            width = 400;
            height = Math.min(650, vh - 40);
        }}
        
        return {{ width: width, height: height }};
    }}
    
    var dims = getResponsiveDimensions();
    
    var container = document.createElement('div');
    container.id = 'cafs-chatbot-container';
    container.style.cssText = 'position:fixed;bottom:0;right:0;z-index:999999;background:transparent;';
    
    var iframe = document.createElement('iframe');
    iframe.id = 'cafs-chatbot-iframe';
    iframe.src = '{embed_url}';
    iframe.style.cssText = 'width:' + dims.width + 'px;height:' + dims.height + 'px;border:none;background:transparent;';
    iframe.allow = 'microphone';
    iframe.title = 'CAFS Chatbot';
    
    container.appendChild(iframe);
    document.body.appendChild(container);
    
    // Update dimensions on resize
    var resizeTimeout;
    window.addEventListener('resize', function() {{
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(function() {{
            var newDims = getResponsiveDimensions();
            var iframeEl = document.getElementById('cafs-chatbot-iframe');
            if (iframeEl) {{
                iframeEl.style.width = newDims.width + 'px';
                iframeEl.style.height = newDims.height + 'px';
            }}
        }}, 150);
    }});
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

