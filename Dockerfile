FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# non-root user
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# 1. dependencies first (cached until requirements-serve.txt changes)
COPY requirements_serve.txt .
RUN pip install -r requirements_serve.txt

# 2. model artifact (changes rarely)
COPY models/champion ./models/champion

# 3. source code (changes often, so it goes last)
COPY app ./app
COPY run.py .

RUN mkdir -p /app/instance && chown -R appuser:appuser /app
USER appuser

ENV MODEL_DIR=/app/models/champion
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5000/health').status==200 else 1)"

CMD ["python", "run.py"]