"""Retrieve -> (rerank) -> answer with citations, or refuse."""

import os
import re
import time
from dataclasses import asdict

from . import store
from .corpus import extract_filters

ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-sonnet-5-5")
PRICES = {"claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-5-5": (0.10, 0.50), "claude-opus-5-5": (4.00, 20.00)}  # USD/MTok
MODES = ("keyword", "dense", "hybrid", "hybrid_rerank")
CANDIDATES, CONTEXT_DOCS, RRF_K = 50, 6, 60
# Mock answerer refuses below this cross-encoder score. 0 = refuse only when nothing is retrieved or the vehicle is unknown (not tuned).
REFUSE_BELOW = float(os.getenv("REFUSE_BELOW", "0"))
NOT_IN_SOURCES = "NOT_IN_SOURCES"
CITE = re.compile(r"\b(?:recall|complaint|fuel):[A-Za-z0-9]+")

SYSTEM = f"""You answer car owners' questions using ONLY the records provided: NHTSA safety recalls, NHTSA owner \
complaints, and EPA fuel-economy listings.

Rules:
- Cite the record id in square brackets after every claim, e.g. [recall:20V314000]. Cite only ids you were given.
- Owner complaints are individual unverified reports, not confirmed defects. Say so when you rely on them, and say how many \
of the provided records describe the issue rather than generalising.
- Only use records for the vehicle (make, model, year) the question asks about. Records for other vehicles don't count.
- If the records don't answer the question, reply with {NOT_IN_SOURCES} on the first line, then one sentence on what's missing. \
Don't answer from general knowledge.
- Under 150 words. Plain prose, no headings."""


def is_mock() -> bool:
    return os.getenv("MOCK_MODE", "0") == "1"


def rrf(*rankings: list[str], k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank + 1)
    return sorted(scores, key=lambda d: (-scores[d], d))


def retrieve(conn, question: str, mode: str = "hybrid_rerank", use_filters: bool = True, k: int = 10) -> dict:
    """Returns ranked doc ids (+ scores where the mode has a meaningful one), the filters used, and per-stage timings."""
    t, timings = time.perf_counter(), {}
    f = extract_filters(question)
    tag, source = (f.tag, f.source) if use_filters else (None, None)

    def lap(name):
        nonlocal t
        now = time.perf_counter()
        timings[name] = round((now - t) * 1000, 2)
        t = now

    kw = dense = []
    if mode in ("keyword", "hybrid", "hybrid_rerank"):
        kw = [i for i, _ in store.keyword_search(conn, question, CANDIDATES, tag, source)]
        lap("keyword_ms")
    if mode in ("dense", "hybrid", "hybrid_rerank"):
        dense = [i for i, _ in store.dense_search(conn, store.embed_query(question), CANDIDATES, tag, source)]
        lap("dense_ms")
    ranked = kw if mode == "keyword" else dense if mode == "dense" else rrf(kw, dense)
    scores = None
    if mode == "hybrid_rerank" and ranked:
        cands = ranked[:30]
        docs = store.fetch(conn, cands)
        s = list(store.reranker().rerank(question, [store.doc_text(docs[i])[:1500] for i in cands]))
        order = sorted(range(len(cands)), key=lambda j: -s[j])
        ranked, scores = [cands[j] for j in order], [float(s[j]) for j in order]
        lap("rerank_ms")
    return {"ids": ranked[:k], "scores": scores[:k] if scores else None, "filters": asdict(f), "timings": timings}


def format_records(docs: list[dict]) -> str:
    out = []
    for d in docs:
        vehicles = ", ".join(v.replace("|", " ") for v in d["vehicles"][:12]) + (" ..." if len(d["vehicles"]) > 12 else "")
        out.append(
            f'<record id="{d["id"]}" type="{d["source"]}" date="{d["date"]}" vehicles="{vehicles}">\n'
            f"{d['title']}\n{d['text'][:1500]}\n</record>"
        )
    return "\n".join(out)


def mock_answer(question: str, docs: list[dict], top_score: float | None) -> tuple[str, int, int]:
    """Extractive stand-in: first two sentences of the top record. Refusal depends on retrieval alone."""
    if not docs or (top_score is not None and top_score < REFUSE_BELOW):
        text = f"{NOT_IN_SOURCES}\nNo record in the index matches this question closely enough."
    else:
        d = docs[0]
        body = d["text"].split("Consequence:")[0].removeprefix("Summary: ")
        n = 99 if d["source"] == "fuel" else 2  # a fuel listing is short and the MPG comes last
        text = " ".join(re.split(r"(?<=[.!?])\s+", body.strip())[:n]) + f" [{d['id']}]"
    return text, max(1, len(SYSTEM + format_records(docs) + question) // 4), max(1, len(text) // 4)


class Claude:
    def __init__(self, model: str = ANSWER_MODEL):
        import anthropic

        self.client, self.model = anthropic.Anthropic(), model

    def __call__(self, system: str, prompt: str, max_tokens: int = 1024) -> tuple[str, int, int]:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low"},
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        if resp.stop_reason == "refusal":
            text = f"{NOT_IN_SOURCES}\nThe model declined to answer."
        return text, resp.usage.input_tokens, resp.usage.output_tokens


def cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p_in, p_out = PRICES[model]
    return (input_tokens * p_in + output_tokens * p_out) / 1e6


def ask(conn, question: str, llm=None, mode: str = "hybrid_rerank", use_filters: bool = True) -> dict:
    t0 = time.perf_counter()
    r = retrieve(conn, question, mode, use_filters, k=CONTEXT_DOCS)
    by_id = store.fetch(conn, r["ids"])
    docs = [by_id[i] for i in r["ids"]]
    top = r["scores"][0] if r["scores"] else None
    if r["filters"]["unknown_vehicle"] or not docs:
        # nothing about this vehicle is indexed: refuse without paying for a model call
        text, i, o, model = f"{NOT_IN_SOURCES}\nThe index has no records for that vehicle.", 0, 0, None
    elif llm is None:
        (text, i, o), model = mock_answer(question, docs, top), "mock"
    else:
        prompt = f"<records>\n{format_records(docs)}\n</records>\n\nQuestion: {question}"
        (text, i, o), model = llm(SYSTEM, prompt), llm.model
    t1 = time.perf_counter()
    refused = text.lstrip().startswith(NOT_IN_SOURCES)
    cited = list(dict.fromkeys(CITE.findall(text)))
    return {
        "question": question,
        "answer": text.removeprefix(NOT_IN_SOURCES).strip() if refused else text.strip(),
        "refused": refused,
        "citations": [c for c in cited if c in by_id],
        "invalid_citations": [c for c in cited if c not in by_id],
        "docs": docs,
        "scores": r["scores"],
        "filters": r["filters"],
        "timings": {**r["timings"], "total_ms": round((t1 - t0) * 1000, 1)},
        "model": model,
        "input_tokens": i,
        "output_tokens": o,
        "cost_usd": cost(model, i, o) if model in PRICES else 0.0,
    }
