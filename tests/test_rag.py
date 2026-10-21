import os

import pytest

os.environ["MOCK_MODE"] = "1"

from carrag.corpus import extract_filters, load_corpus, tags  # noqa: E402
from carrag.rag import NOT_IN_SOURCES, mock_answer, rrf  # noqa: E402
from eval.run_eval import metrics  # noqa: E402


@pytest.mark.parametrize(
    "q,tag,source,unknown",
    [
        ("Are there airbag recalls on a 2018 Honda Accord?", "HONDA|ACCORD|2018", "recall", False),
        ("my 2019 crv has an oil smell, anyone else report this?", "HONDA|CR-V|2019", "complaint", False),
        ("What's the combined mpg of the 2020 F150?", "FORD|F-150|2020", "fuel", False),
        ("2017 Chevy Silverado transmission recall", "CHEVROLET|SILVERADO 1500|2017", "recall", False),
        ("Is the 2021 Ram 1500 recalled for anything?", "RAM|1500|2021", "recall", False),
        ("range of a 2022 Tesla Model 3", "TESLA|MODEL 3|2022", "fuel", False),
        ("Toyota recalls", "TOYOTA", "recall", False),
        ("Is there a recall on the 2017 BMW X5?", None, "recall", True),
        ("2009 Toyota Camry recall", "TOYOTA|CAMRY|2009", "recall", True),
        # two hints (recall + complaint wording) -> no source filter
        ("Is there a recall for the brake problem other owners report on the 2016 Focus?", "FORD|FOCUS|2016", None, False),
    ],
)
def test_extract_filters(q, tag, source, unknown):
    f = extract_filters(q)
    assert (f.tag, f.source, f.unknown_vehicle) == (tag, source, unknown)


def test_tags_cover_every_level():
    d = {"vehicles": ["HONDA|ACCORD|2018", "HONDA|CIVIC|2019"]}
    assert tags(d) == ["HONDA", "HONDA|ACCORD", "HONDA|ACCORD|2018", "HONDA|CIVIC", "HONDA|CIVIC|2019"]


def test_corpus_is_scrubbed():
    import re

    vin = re.compile(r"\b(?=[A-HJ-NPR-Z0-9]*\d)(?=[A-HJ-NPR-Z0-9]*[A-Z])[A-HJ-NPR-Z0-9]{17}\b")
    email = re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", re.I)
    docs = load_corpus()
    assert len(docs) > 8000 and {d["source"] for d in docs} == {"recall", "complaint", "fuel"}
    # owner complaints are scrubbed; recalls keep manufacturers' public contact addresses
    assert not [d["id"] for d in docs if d["source"] == "complaint" and (vin.search(d["text"]) or email.search(d["text"]))]


def test_rrf():
    assert rrf(["a", "b", "c"], ["c", "a"]) == ["a", "c", "b"]
    assert rrf([], []) == []


def test_metrics():
    m = metrics(["x", "g", "y"], {"g"})
    assert m["hit@1"] is False and m["hit@5"] and m["mrr@10"] == 0.5
    assert metrics(["g"], {"g"})["ndcg@10"] == 1.0
    assert metrics(["x"], {"g"})["mrr@10"] == 0.0


def test_mock_answer_cites_or_refuses(monkeypatch):
    doc = {
        "id": "recall:20V314000",
        "source": "recall",
        "date": "",
        "vehicles": ["HONDA|ACCORD|2018"],
        "title": "t",
        "text": "Summary: The fuel pump may fail. It can stall.\nConsequence: crash.",
    }
    text, _, _ = mock_answer("q", [doc], 3.0)
    assert text == "The fuel pump may fail. It can stall. [recall:20V314000]"
    assert mock_answer("q", [], None)[0].startswith(NOT_IN_SOURCES)
    monkeypatch.setattr("carrag.rag.REFUSE_BELOW", 5.0)
    assert mock_answer("q", [doc], 3.0)[0].startswith(NOT_IN_SOURCES)


needs_db = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="needs Postgres (python -m carrag.seed)")


@pytest.fixture(scope="module")
def conn():
    from carrag import seed, store

    seed.main([])
    return store.connect()


@needs_db
def test_hybrid_rerank_finds_known_recall(conn):
    from carrag.rag import ask

    r = ask(conn, "Is there a recall on the fuel pump of my 2018 Honda Accord?")
    assert "recall:20V314000" in [d["id"] for d in r["docs"]][:3]
    assert r["citations"] and not r["invalid_citations"] and not r["refused"]
    assert all("HONDA|ACCORD|2018" in d["vehicles"] for d in r["docs"])  # filter applied


@needs_db
def test_unknown_vehicle_refuses_without_model_call(conn):
    from carrag.rag import ask

    r = ask(conn, "Is there a recall on the 2017 BMW X5?", llm=object())  # llm must not be called
    assert r["refused"] and r["model"] is None and r["cost_usd"] == 0.0


@needs_db
def test_api(conn):
    from fastapi.testclient import TestClient

    from carrag.app import create_app

    c = TestClient(create_app())
    assert sum(c.get("/api/stats").json()["docs"].values()) == len(load_corpus())
    r = c.post("/api/ask", json={"question": "What is the combined MPG of a 2019 Toyota Camry?", "mode": "hybrid"}).json()
    assert r["docs"] and all(d["source"] == "fuel" for d in r["docs"])
    assert c.post("/api/ask", json={"question": "x", "mode": "bogus"}).status_code == 422
