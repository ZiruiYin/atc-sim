# Hugging Face Space (Docker) image for the FULL simulator — Flask backend +
# AUTO planner (torch). The GitHub Pages build does NOT use this; it runs the
# sim in-browser via Pyodide and has no AUTO mode. See DEPLOY.md.
#
# HF free CPU Spaces serve on port 7860. torch is installed from the CPU wheel
# index so the image stays small (no multi-GB CUDA payload) and runs on the
# free CPU tier.
FROM python:3.11-slim

# HF runs the container as a non-root user (uid 1000). Set up a writable home.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python deps first for layer caching. CPU-only torch keeps this lean.
COPY requirements.txt .
RUN pip install --no-cache-dir flask>=2.0 waitress>=3.0 numpy>=1.24 \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir piper-tts \
 && pip install --no-cache-dir faster-whisper transformers   # voice: STT + text->DSL parser

# App source (the .dockerignore trims training/dev cruft).
COPY . .
RUN chown -R user:user /app
USER user

# Pre-load the AUTO policy + worker pool at server boot so the first time a user
# engages AUTO it's instant (no cold torch import / checkpoint load / process
# spawn). See app.py:_warm_auto. Lazy (unset) for local runs.
ENV ATC_WARM_AUTO=1

EXPOSE 7860
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "7860"]
