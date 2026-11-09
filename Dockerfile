# Imagem do pipeline do Tema 9.
# Base CUDA 12.4 + cuDNN 9 (o faster-whisper usa CTranslate2 com GPU); roda também em CPU.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}"

# FFmpeg (áudio e keyframes), Tesseract com o idioma português (OCR)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        tesseract-ocr \
        tesseract-ocr-por \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv fixado na versão usada localmente
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv

WORKDIR /app

# Dependências primeiro (camada em cache enquanto o lock não mudar)
COPY .python-version pyproject.toml uv.lock ./
RUN uv python install 3.12 \
    && uv sync --frozen --no-install-project

# Código e testes
COPY src ./src
COPY tests ./tests
ENV PYTHONPATH=/app/src

CMD ["pytest", "-q"]
