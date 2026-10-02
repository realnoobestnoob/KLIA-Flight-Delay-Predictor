# syntax=docker/dockerfile:1
FROM python:3.11-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /w
COPY requirements-serve.txt .
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r requirements-serve.txt

FROM python:3.11-slim
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    KLIA_CACHE_DIR=/tmp/klia-cache KLIA_ARTIFACTS_DIR=/tmp/klia-artifacts PORT=8000
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY klia ./klia
COPY config ./config
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','8000'), timeout=4)"
# One worker on purpose: the model lives in this process's memory.
CMD ["sh", "-c", "exec uvicorn klia.api.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
