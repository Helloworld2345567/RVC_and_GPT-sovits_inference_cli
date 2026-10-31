ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu129

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/GPT-SoVITS \
    NLTK_DATA=/opt/nltk_data

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt constraints.txt ./
COPY RVC/requirements-inference.txt ./RVC/requirements-inference.txt
COPY GPT-SoVITS/requirements.txt ./GPT-SoVITS/requirements.txt
COPY scripts/prepare_language_resources.py ./scripts/prepare_language_resources.py
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 --index-url ${TORCH_INDEX_URL} \
    && python -m pip install --no-cache-dir -r requirements.txt -c constraints.txt \
    && python scripts/prepare_language_resources.py

COPY RVC ./RVC
COPY GPT-SoVITS ./GPT-SoVITS
RUN python -m pip install --no-cache-dir --no-deps -e ./RVC
WORKDIR /app/GPT-SoVITS
EXPOSE 9880

ENTRYPOINT ["python", "api_v2.py"]
CMD ["-a", "0.0.0.0", "-p", "9880", "-c", "GPT_SoVITS/configs/tts_local.yaml"]
