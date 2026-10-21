FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY carrag ./carrag
RUN pip install --no-cache-dir .
# Bake the embedding + rerank models into the image so containers start offline.
ENV FASTEMBED_CACHE_PATH=/models HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
RUN python -c "from carrag import store; store.embedder(); store.reranker()"
COPY data ./data
EXPOSE 8000
CMD ["uvicorn", "carrag.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
