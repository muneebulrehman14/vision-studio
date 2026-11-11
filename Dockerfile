# Full-featured image for any container host (Render, Railway, Fly.io, HF Spaces).
# Includes TensorFlow, so object prediction works.
#
#   docker build -t vision-studio .
#   docker run -p 7860:7860 vision-studio

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    GRADIO_TEMP_DIR=/tmp/gradio \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# System libraries Pillow and TensorFlow need at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-tf.txt ./
RUN pip install -r requirements.txt -r requirements-tf.txt

COPY . .

EXPOSE 7860

# uvicorn serves the ASGI entrypoint (app.py) — no browser auto-open in a container.
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
