# AniLogic application image (Python 3.12). The image contains code only: data, artifacts and the
# business database live in mounted volumes and are produced by the pipeline commands.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 UV_SYSTEM_PYTHON=1
WORKDIR /srv/anilogic
RUN pip install uv==0.12.5
COPY pyproject.toml uv.lock README.md ./
COPY app ./app
COPY pipelines ./pipelines
COPY evaluation ./evaluation
COPY configs ./configs
COPY migrations ./migrations
COPY alembic.ini ./
# ml extra = torch + sentence-transformers (BGE retrieval, NCF); set EXTRAS="" for a slim base image
ARG EXTRAS="--extra ml"
RUN uv export --frozen --no-hashes ${EXTRAS} --no-dev -o /tmp/requirements.txt \
 && pip install -r /tmp/requirements.txt && pip install --no-deps -e .
RUN useradd --create-home anilogic && mkdir -p data artifacts var evaluation/runs && chown -R anilogic /srv/anilogic
USER anilogic
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/system/health').status==200 else 1)"
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
