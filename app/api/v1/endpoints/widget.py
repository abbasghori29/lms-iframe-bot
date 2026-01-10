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
    Fixes mobile issues: prevents parent page scrollbars, proper fullscreen handling.
    """
    scheme, host = get_base_url(request)
    embed_url = f"{scheme}://{host}/api/v1/widget/embed"
    
    script = f"""
(function() {{
    // CAFS Chatbot Embed Script - Mobile-Optimized Version
    if (document.getElementById('cafs-chatbot-container')) return;
    
    var isMobile = function() {{
        return window.innerWidth <= 768;
    }};
    
    // Calculate responsive dimensions
    function getResponsiveDimensions() {{
        var vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
        var vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);
        
        var width, height;
        
        // Mobile: Full screen but container stays small until opened
        if (vw <= 768) {{
            // Container needs to be larger to fit the shadow without clipping
            // 60px button + ~30px shadow/glow = ~120px safe area
            width = 150; 
            height = 150;
        }} else if (vw <= 1024) {{
            width = 400;
            height = Math.min(680, vh - 60);
        }} else {{
            width = 420;
            height = Math.min(720, vh - 60);
        }}
        
        return {{ width: width, height: height, isMobile: vw <= 768 }};
    }}
    
    var dims = getResponsiveDimensions();
    
    // Create container with proper margins from edges
    var container = document.createElement('div');
    container.id = 'cafs-chatbot-container';
    
    // Default style - positioned at bottom right corner to allow shadows to flow up/left
    // pointer-events: none allows clicks to pass through the empty transparent parts of the container
    container.style.cssText = 'position:fixed;bottom:0px;right:0px;z-index:999999;background:transparent;pointer-events:none;';
    
    if (dims.isMobile) {{
        // On mobile, container is small - iframe inside handles full screen when opened
        container.style.width = dims.width + 'px';
        container.style.height = dims.height + 'px';
    }} else {{
        // Desktop positioning adjustments
        container.style.width = dims.width + 'px';
        container.style.height = dims.height + 'px';
        // Check if we need to adjust for desktop margins if 0,0 positioning is used
        // Actually for desktop we might want to keep it slightly offset if the shadow is huge
        // But 0,0 is safest for shadows if everything is inside.
    }}
    
    var iframe = document.createElement('iframe');
    iframe.id = 'cafs-chatbot-iframe';
    iframe.src = '{embed_url}';
    // pointer-events: auto re-enables clicks on the iframe itself (button/window)
    iframe.style.cssText = 'width:100%;height:100%;border:none;background:transparent;pointer-events:auto;';
    iframe.allow = 'microphone';
    iframe.title = 'CAFS Chatbot';
    
    container.appendChild(iframe);
    document.body.appendChild(container);
    
    // Store original body styles for restoration
    var originalBodyStyles = null;
    
    // Listen for messages from iframe to handle mobile full screen
    window.addEventListener('message', function(event) {{
        if (event.data && event.data.type === 'cafsChatOpen') {{
            if (isMobile()) {{
                // Store original body styles
                originalBodyStyles = {{
                    overflow: document.body.style.overflow
                }};
                
                // Prevent body scroll - just overflow hidden is usually enough and smoother than position:fixed
                document.body.style.overflow = 'hidden';
                
                // Expand container to full screen
                container.style.width = '100%';
                container.style.height = '100%';
                container.style.pointerEvents = 'auto'; // Enable clicks everywhere in the container (overlay)
            }}
        }}
        if (event.data && event.data.type === 'cafsChatClose') {{
            if (isMobile()) {{
                // Restore body scroll
                if (originalBodyStyles) {{
                    document.body.style.overflow = originalBodyStyles.overflow || '';
                }}
                
                // Shrink container back to toggle size
                var newDims = getResponsiveDimensions();
                container.style.width = newDims.width + 'px';
                container.style.height = newDims.height + 'px';
                container.style.pointerEvents = 'none'; // Back to pass-through
            }}
        }}
    }});
    
    // Handle resize events
    var resizeTimeout;
    window.addEventListener('resize', function() {{
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(function() {{
            var newDims = getResponsiveDimensions();
            var containerEl = document.getElementById('cafs-chatbot-container');
            // Only resize if not currently open on mobile
            // Use a flag check or check if we are in fullscreen mode (width is '100%')
            var isCurrentlyFullScreen = containerEl.style.width === '100%' && isMobile();
            
            if (containerEl && !isCurrentlyFullScreen) {{
                containerEl.style.width = newDims.width + 'px';
                containerEl.style.height = newDims.height + 'px';
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

