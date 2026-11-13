# syntax=docker/dockerfile:1
FROM python:3.11.14-slim-bookworm
WORKDIR /opt/shorts
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HOME=/data/models/huggingface \
    OLLAMA_MODELS=/data/models/ollama COMFYUI_MODELS=/data/models/comfyui \
    OUTPUT_DIR=/data/outputs MUSIC_DIR=/data/assets/music GRADIO_ANALYTICS_ENABLED=False
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::https::Timeout=30 -o Acquire::Retries=2 update \
    && apt-get -o Acquire::https::Timeout=30 -o Acquire::Retries=2 install -y --no-install-recommends ffmpeg fonts-dejavu-core git curl ca-certificates libsndfile1 sox libgomp1 \
    && rm -rf /var/lib/apt/lists/*
ARG COMFY_COMMIT
RUN test -n "$COMFY_COMMIT" && git clone --filter=blob:none https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI \
    && git -C /opt/ComfyUI checkout "$COMFY_COMMIT" && rm -rf /opt/ComfyUI/.git
ARG OLLAMA_VERSION=0.13.5
ARG OLLAMA_SHA256=41fb93ff8be35e4d2d22bafd1c42b487efb15b766076d976766bd1ee4db3f8e2
RUN curl -fL --connect-timeout 30 --max-time 1800 --retry 3 "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-amd64.tgz" -o /tmp/ollama.tgz \
    && echo "$OLLAMA_SHA256  /tmp/ollama.tgz" | sha256sum -c - \
    && tar -xzf /tmp/ollama.tgz -C /usr && rm /tmp/ollama.tgz
COPY requirements.lock requirements-whisperx.txt ./
RUN pip install --no-cache-dir -r requirements.lock && pip install --no-cache-dir --no-deps -r requirements-whisperx.txt
RUN python -c "from whisperx.alignment import load_align_model; from qwen_tts import Qwen3TTSModel"
RUN useradd --create-home --uid 1000 shorts && mkdir -p /data && chown shorts:shorts /data /opt/ComfyUI/models /opt/ComfyUI/output /opt/ComfyUI/input /opt/ComfyUI
COPY app ./app
COPY config ./config
COPY workflows ./workflows
COPY models.lock.yaml start.sh ./
RUN chmod +x start.sh
USER shorts
EXPOSE 7860
ENTRYPOINT ["./start.sh"]
CMD ["python", "-m", "app.ui"]
