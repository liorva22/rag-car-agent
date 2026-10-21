"""FastAPI: POST /api/ask, GET /api/stats, and the chat page at /."""

import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import store
from .rag import MODES, Claude, ask, is_mock


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: str = Field("hybrid_rerank", pattern="|".join(MODES))
    use_filters: bool = True


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Car Agent")
    conn = store.connect()
    lock = threading.Lock()  # ponytail: one shared connection; use a psycopg_pool pool if concurrent users matter
    llm = None if is_mock() else Claude()

    @app.post("/api/ask")
    def api_ask(req: Ask) -> dict:
        with lock:
            r = ask(conn, req.question, llm, req.mode, req.use_filters)
        r["docs"] = [{k: d[k] for k in ("id", "source", "title", "date", "url", "text")} for d in r["docs"]]
        return r

    @app.get("/api/stats")
    def stats() -> dict:
        with lock:
            rows = conn.execute("SELECT source, count(*) FROM docs GROUP BY source ORDER BY source").fetchall()
        return {"docs": dict(rows), "mode": "mock" if is_mock() else "live"}

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    return app
