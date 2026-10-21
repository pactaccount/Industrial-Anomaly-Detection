# Argus industrial inspection API + console.
#
# Multi-stage: build the React console with Node, then serve it from the
# Python API so the whole product is a single process and a single origin.

FROM node:22-slim AS console
WORKDIR /console
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.11-slim AS runtime

# OpenCV needs these; torch pulls in its own.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY argus/ ./argus/
COPY scripts/ ./scripts/
COPY --from=console /console/dist ./web/dist

# Model artefacts, the database and inspection media live here.
ENV ARGUS_DATA_DIR=/data
VOLUME ["/data"]

ENV PYTHONUNBUFFERED=1 \
    ARGUS_CORS_ORIGINS=""

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "argus.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
