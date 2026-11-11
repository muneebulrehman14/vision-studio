"""ASGI entrypoint — used by Vercel, Docker, and any ASGI server.

Vercel's Python runtime looks for a top-level `app` in `app.py` (or `src/app.py`)
and runs it as a serverless function. Locally you would normally use
`python src/vision_web.py`, but `uvicorn app:app` works too:

    uvicorn app:app --port 7860

The Gradio UI is mounted at "/" on a FastAPI application, so every route
(`/`, `/gradio_api/...`, static assets) is served from one ASGI object.
"""

import os
import tempfile

# Serverless filesystems are read-only apart from /tmp; point Gradio's cache
# and upload staging at a writable directory before it is imported.
os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(tempfile.gettempdir(), "gradio"))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

try:  # when deployed, the repo root is the working directory
    from src.vision_web import create_app
except ImportError:  # pragma: no cover - fallback when run from inside src/
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.vision_web import create_app

import gradio as gr
from fastapi import FastAPI

demo = create_app()
app = gr.mount_gradio_app(FastAPI(title="Vision Studio"), demo, path="/")
