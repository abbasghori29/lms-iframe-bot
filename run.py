"""
Development server runner
"""
import os
import sys
import uvicorn

if __name__ == "__main__":
    # Ensure Windows consoles with cp1252 don't crash on Unicode log output.
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8005,
        reload=True,
        log_level="info",
        access_log=True,
    )

