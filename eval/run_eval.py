"""Evaluate retrieval (offline) and end-to-end answers (mock or live) on eval/questions.json.

python eval/run_eval.py retrieval                  # 4 retrieval modes x filters on/off -> results/retrieval.json
MOCK_MODE=1 python eval/run_eval.py answers        # extractive mock answers -> results/answers_mock.json
python eval/run_eval.py answers                    # Claude answers + Haiku judge -> results/answers_live.json
"""

import argparse
import json
import math
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from carrag import store  # noqa: E402
from carrag.corpus import load_corpus  # noqa: E402
from carrag.rag import MODES, Claude, ask, cost, format_records, is_mock, retrieve  # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.json"
RESULTS = ROOT / "results"
JUDGE_MODEL = "claude-haiku-5-5"
MPG = re.compile(r"EPA fuel economy: (\d+) city / (\d+) highway / (\d+) combined")

JUDGE = """You grade a car-owner assistant's answer. You get the question, a reference answer written from the record \
that answers it, the records the assistant was shown, and the assistant's answer.

- correct: true if the answer states the key facts of the reference answer (the defect / symptom / remedy, or the \
MPG numbers). Extra correct detail is fine. A refusal is not correct.
- faithful: true if every factual claim in the answer is supported by the records shown. Ignore style.

Reply with JSON only: {"correct": true|false, "faithful": true|false, "notes": "<= 20 words"}"""


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]


def relevant_sets(questions: list[dict]) -> dict[str, set[str]]:
    """Gold record, plus fuel listings for the same vehicle-year with identical EPA numbers (same answer, other trim)."""
    docs = {d["id"]: d for d in load_corpus()}
    fuel_by_key: dict[tuple, set[str]] = {}
    for d in docs.values():
        if d["source"] == "fuel" and (m := MPG.search(d["text"])):
            fuel_by_key.setdefault((d["vehicles"][0], m.groups()), set()).add(d["id"])
    rel = {}
    for q in questions:
        if not q["answerable"]:
            continue
        g = docs[q["gold"]]
        m = MPG.search(g["text"]) if g["source"] == "fuel" else None
        rel[q["id"]] = fuel_by_key[(g["vehicles"][0], m.groups())] if m else {g["id"]}
    return rel


def metrics(ranked: list[str], rel: set[str]) -> dict:
    hits = [i in rel for i in ranked[:10]]
    first = next((n for n, h in enumerate(hits) if h), None)
    dcg = sum(1 / math.log2(n + 2) for n, h in enumerate(hits) if h)
    ideal = sum(1 / math.log2(n + 2) for n in range(min(len(rel), 10)))
    return {
        "hit@1": hits[:1] == [True],
        "hit@5": any(hits[:5]),
        "hit@10": any(hits),
        "mrr@10": 1 / (first + 1) if first is not None else 0.0,
        "ndcg@10": dcg / ideal,
    }


def mean(xs) -> float:
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def pct(xs, p) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p * len(xs)))], 1) if xs else 0.0


def eval_retrieval(conn) -> dict:
    qs = [q for q in load_questions() if q["answerable"]]
    rel = relevant_sets(qs)
    store.reranker(), store.embed_query("warm up")  # load models before timing
    out = {"n_questions": len(qs), "configs": {}}
    for use_filters in (False, True):
        for mode in MODES:
            rows = []
            for q in qs:
                t = time.perf_counter()
                r = retrieve(conn, q["question"], mode, use_filters, k=10)
                rows.append({**metrics(r["ids"], rel[q["id"]]), "ms": (time.perf_counter() - t) * 1000, "source": q["source"]})
            name = f"{mode}{'+filters' if use_filters else ''}"
            out["configs"][name] = {
                **{m: mean(r[m] for r in rows) for m in ("hit@1", "hit@5", "hit@10", "mrr@10", "ndcg@10")},
                "p50_ms": pct([r["ms"] for r in rows], 0.5),
                "p95_ms": pct([r["ms"] for r in rows], 0.95),
                "hit@5_by_source": {
                    s: mean(r["hit@5"] for r in rows if r["source"] == s) for s in ("recall", "complaint", "fuel")
                },
            }
            print(name, out["configs"][name])
    filt = [retrieve(conn, q["question"], "keyword", True)["filters"] for q in qs]
    out["filter_extraction"] = {
        "vehicle_tag_found": mean(f["tag"] is not None for f in filt),
        "tag_matches_gold_vehicle": mean(f["tag"] == q["vehicle"] for f, q in zip(filt, qs, strict=True)),
        "source_hint_found": mean(f["source"] is not None for f in filt),
        "source_hint_correct_when_found": mean(f["source"] == q["source"] for f, q in zip(filt, qs, strict=True) if f["source"]),
    }
    return out


def eval_answers(n_workers: int = 6) -> dict:
    qs = load_questions()
    rel = relevant_sets(qs)
    llm = None if is_mock() else Claude()
    judge = None if is_mock() else Claude(JUDGE_MODEL)
    local = threading.local()
    store.reranker(), store.embed_query("warm up")

    def run(q):
        if not hasattr(local, "conn"):
            local.conn = store.connect()
        r = ask(local.conn, q["question"], llm)
        row = {
            k: r[k]
            for k in (
                "answer",
                "refused",
                "citations",
                "invalid_citations",
                "timings",
                "model",
                "input_tokens",
                "output_tokens",
                "cost_usd",
            )
        }
        row |= {"id": q["id"], "question": q["question"], "answerable": q["answerable"], "context": [d["id"] for d in r["docs"]]}
        if q["answerable"]:
            g = rel[q["id"]]
            row |= {
                "source": q["source"],
                "gold_in_context": bool(g & set(row["context"])),
                "gold_cited": bool(g & set(r["citations"])),
            }
            if judge and not r["refused"]:
                prompt = (
                    f"Question: {q['question']}\n\nReference answer: {q['reference']}\n\n<records>\n{format_records(r['docs'])}\n"
                    f"</records>\n\nAssistant answer:\n{r['answer']}"
                )
                text, i, o = judge(JUDGE, prompt, max_tokens=300)
                # the verdicts come before the notes, so they survive a note that breaks the JSON (stray quote, cut-off)
                v = {k: b == "true" for k in ("correct", "faithful") if (b := (re.search(rf'"{k}":\s*(true|false)', text) or [0, ""])[1])}
                row["judge_cost_usd"] = cost(JUDGE_MODEL, i, o)
                if "correct" in v and "faithful" in v:
                    row |= {"correct": bool(v["correct"]), "faithful": bool(v["faithful"]), "judge_notes": (re.search(r'"notes":\s*"(.*)', text) or [0, ""])[1][:200]}
                else:
                    row["judge_unparsed"] = text[:200]
        else:
            row["kind"] = q["kind"]
        return row

    with ThreadPoolExecutor(1 if is_mock() else n_workers) as ex:
        rows = list(ex.map(run, qs))

    ans, una = [r for r in rows if r["answerable"]], [r for r in rows if not r["answerable"]]
    graded = [r for r in ans if "correct" in r]
    summary = {
        "mode": "mock" if is_mock() else "live",
        "answer_model": "mock-extractive" if is_mock() else llm.model,
        "n_answerable": len(ans),
        "n_unanswerable": len(una),
        "answered": mean(not r["refused"] for r in ans),
        "false_refusal_rate": mean(r["refused"] for r in ans),
        "gold_in_context": mean(r["gold_in_context"] for r in ans),
        "gold_cited": mean(r["gold_cited"] for r in ans),
        "answers_with_citation": mean(bool(r["citations"]) for r in ans if not r["refused"]),
        "invalid_citation_answers": sum(bool(r["invalid_citations"]) for r in rows),
        "correct_refusal_rate": mean(r["refused"] for r in una),
        "correct_refusal_by_kind": {
            k: mean(r["refused"] for r in una if r["kind"] == k) for k in sorted({r["kind"] for r in una})
        },
        "p50_ms": pct([r["timings"]["total_ms"] for r in rows], 0.5),
        "p95_ms": pct([r["timings"]["total_ms"] for r in rows], 0.95),
        "answer_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
        "judge_cost_usd": round(sum(r.get("judge_cost_usd", 0) for r in rows), 4),
        "cost_per_query_usd": round(statistics.mean(r["cost_usd"] for r in rows), 5),
    }
    if graded:
        summary |= {
            "n_judged": len(graded),
            "judge_unparsed": sum("judge_unparsed" in r for r in ans),
            "correct_of_answerable": round(sum(r["correct"] for r in graded) / len(ans), 4),
            "correct_when_answered": mean(r["correct"] for r in graded),
            "faithful_when_answered": mean(r["faithful"] for r in graded),
            "correct_by_source": {
                s: mean(r["correct"] for r in graded if r["source"] == s) for s in ("recall", "complaint", "fuel")
            },
        }
    print(json.dumps(summary, indent=2))
    return {"summary": summary, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["retrieval", "answers"])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    RESULTS.mkdir(exist_ok=True)
    if args.what == "retrieval":
        report = eval_retrieval(store.connect())
        out = args.out or RESULTS / "retrieval.json"
    else:
        report = eval_answers()
        out = args.out or RESULTS / f"answers_{report['summary']['mode']}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
