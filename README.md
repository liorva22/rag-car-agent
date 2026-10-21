# RAG Car Agent

**Answers car owners' questions from NHTSA recalls, NHTSA owner complaints and EPA fuel-economy listings. Every claim carries a citation, and when the index doesn't cover the question it refuses instead of guessing.**

[![CI](https://github.com/liorva22/rag-car-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/liorva22/rag-car-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)

## Demo

![Chat UI](docs/img/chat_ui.png)

*A real `docker compose up` run in `MOCK_MODE=1`: a question about a 2016 Ford Focus. The vehicle filter was extracted from the question, the cited record is highlighted, and each retrieved record shows its cross-encoder score and links to the source.*

## The problem

"Is there a recall on my car?" and "do other owners see this problem?" are answered by public data that nobody reads. A plain LLM answers from memory and invents campaign numbers. This agent answers **only** from retrieved records, cites each one by id, says so when a claim rests on unverified owner complaints, and replies `NOT_IN_SOURCES` when the records don't answer.

## Architecture

```mermaid
flowchart LR
    q([Owner question]) --> f[Rule-based filters<br/>make / model / year / record type]
    f -->|unknown vehicle| refuse1[Refuse: NOT_IN_SOURCES<br/>no model call]
    f --> kw[Postgres full-text<br/>tsvector, top 50]
    f --> dn[pgvector cosine<br/>bge-small, top 50]
    kw --> rrf[Reciprocal rank fusion]
    dn --> rrf
    rrf --> rr[Cross-encoder rerank<br/>MiniLM, top 30 to 6]
    rr --> ans{Answerer}
    ans -->|live| claude[(claude-sonnet-5-5<br/>cited answer or NOT_IN_SOURCES)]
    ans -.->|MOCK_MODE=1| mock[Extractive mock<br/>first sentences of top record]
    claude --> out([Answer + citations<br/>validated against retrieved ids])
    mock --> out
    pg[(Postgres + pgvector<br/>9,168 docs: recalls, complaints, EPA)] --- kw
    pg --- dn
```

- **Corpus.** 9,168 records from public-domain US government sources, no key needed: 827 NHTSA recall campaigns, 6,183 owner complaints (a seeded sample of 20 per vehicle-year; VINs and contact details scrubbed) and 2,158 EPA fuel-economy configurations. They cover 34 models (Toyota, Honda, Ford, Chevrolet, Nissan, Hyundai, Kia, Jeep, Subaru, Tesla, Ram, Mazda), model years 2015-2024.
- **Retrieval.** One Postgres table serves both search types: a `tsvector` index with OR-ed query terms for keywords, and an exact pgvector cosine scan (BAAI/bge-small, 384-d, computed locally with fastembed). The two rankings are fused with RRF and reranked by a local cross-encoder.
- **Filters.** Make, model, year and record type are extracted from the question with rules (aliases such as "CRV", "F150", "Chevy"). They narrow every search through a GIN-indexed tag array. A question naming a year or make that isn't indexed is refused with no model call.
- **Answer.** Claude gets the top 6 records and a system prompt that requires a record id after every claim, flags complaints as unverified, ignores records for other vehicles, and demands `NOT_IN_SOURCES` otherwise. Citations in the output are checked against the ids that were actually retrieved.

## Key results

Evaluated on the 200 questions in [`eval/questions.json`](eval/questions.json) (80 recall, 70 complaint, 50 fuel). Each was written by Claude from one sampled record, which is that question's gold record. **The questions were written by an AI model, not by car owners**, and the generator saw the gold record, so questions can echo its wording. Treat absolute numbers as optimistic.

### Retrieval (offline, no API; [`results/retrieval.json`](results/retrieval.json))

`python eval/run_eval.py retrieval` on the 200 answerable questions. Hit@k = a relevant record is in the top k. For fuel questions, other trims of the same vehicle-year with identical EPA numbers count as relevant. Latencies are CPU, one Windows laptop, and include the filter step.

| Config | hit@1 | hit@5 | hit@10 | MRR@10 | nDCG@10 | p50 latency |
|---|---|---|---|---|---|---|
| keyword (no filters) | 0.270 | 0.650 | 0.780 | 0.420 | 0.497 | 213 ms |
| dense (no filters) | 0.585 | 0.860 | 0.930 | 0.699 | 0.741 | 44 ms |
| hybrid (no filters) | 0.660 | 0.910 | 0.960 | 0.766 | 0.798 | 309 ms |
| hybrid + rerank (no filters) | 0.790 | 0.970 | 0.985 | 0.868 | 0.887 | 2,125 ms |
| keyword + filters | 0.715 | 0.965 | 0.985 | 0.823 | 0.853 | 8 ms |
| dense + filters | 0.745 | 0.970 | 1.000 | 0.839 | 0.868 | 17 ms |
| hybrid + filters | 0.770 | 0.985 | 1.000 | 0.862 | 0.885 | 25 ms |
| **hybrid + rerank + filters** | **0.860** | **0.990** | 0.995 | **0.918** | **0.929** | 1,032 ms |

- **Filters help most.** They lift keyword hit@5 from 0.65 to 0.965, because the vehicle name is the part of a question that keyword search can't weigh.
- **Dense beats keyword without filters** (hit@5 0.86 vs 0.65), mostly on complaints (0.86 vs 0.43), where owners and the question use different words for the same symptom.
- **Reranking buys hit@1** (0.77 to 0.86 with filters) **and costs about 1 second on CPU.** Hit@5 barely moves (0.985 to 0.99). If you only show 5 records, hybrid + filters at 25 ms is the better trade.
- **Filter extraction:** a vehicle was found for 100% of questions and matched the gold vehicle for 99.5%. A record-type hint was found for 98% and was right every time it fired.

### Answers, mock mode (retrieval is real, answer text is extractive; [`results/answers_mock.json`](results/answers_mock.json))

`MOCK_MODE=1 python eval/run_eval.py answers`

| Metric | Value |
|---|---|
| Gold record in the 6 records given to the answerer | 99.5% |
| Answers that cite the gold record | 86.0% |
| Answers with a citation / invalid (made-up) citations | 100% / 0 |
| p50 / p95 latency (retrieval + rerank, mock answer) | 984 ms / 1,471 ms |

**Honest scope.**
- The mock answerer pastes the first two sentences of the top record, so **answer correctness and faithfulness are not measured in mock mode**, only retrieval and citation plumbing.
- **Refusal behavior is not evaluated.** An earlier version of the eval had 30 unanswerable questions. I removed them: the mock only refuses when the vehicle isn't indexed (that path is covered by unit tests), and the live model's refusals can't be measured without API credit. In mock mode an off-topic question about an indexed car ("oil change interval for a 2018 Civic") gets an unrelated complaint pasted back ([screenshot](docs/img/mock_offtopic_miss.png)). `REFUSE_BELOW` defaults to 0 and is untuned.

### Live answers (Claude Sonnet writes, Claude Haiku judges): **not run**

`python eval/run_eval.py answers` is implemented (correctness and faithfulness judged by Haiku against the gold reference, cost per query), but the API credit ran out before a run completed. **There are no live numbers in this repo, and none are claimed.** Spend so far: $0.58 generating the questions (including the 30 removed ones), plus failed live attempts that saved no results.

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Index | Postgres 16 + pgvector, one table | Full-text, vectors and tag filters in one SQL query. Exact scan is a few ms at 9k rows. |
| Embeddings | BAAI/bge-small-en-v1.5 via fastembed (ONNX) | Local, CPU-only, no torch. Anthropic has no embeddings endpoint. Vectors are committed (`data/embeddings.npz`, float16). |
| Reranker | MiniLM cross-encoder via fastembed | Biggest single quality jump on hit@1 in the table above. |
| Answers | Claude Sonnet 5.5 (live) / extractive mock | Citations are required and validated. |
| API | FastAPI + a single static page | `POST /api/ask`, `GET /api/stats`. |

## Quickstart

```bash
git clone https://github.com/liorva22/rag-car-agent && cd rag-car-agent
docker compose up -d --build        # pgvector + one-shot seed + app; MOCK_MODE=1, no keys
open http://localhost:8106          # chat UI (on Linux: xdg-open)
# live answers: MOCK_MODE=0 ANTHROPIC_API_KEY=... docker compose up -d --build
```

Offline eval (needs a running Postgres, e.g. `docker compose up -d db`):

```bash
pip install -e ".[dev]"
export DATABASE_URL=postgresql://car:car@localhost:5438/car MOCK_MODE=1
python -m carrag.seed                         # load the corpus + committed embeddings
python eval/run_eval.py retrieval             # -> results/retrieval.json
python eval/run_eval.py answers               # -> results/answers_mock.json
python scripts/fetch_data.py                  # optional: rebuild the corpus from NHTSA / fueleconomy.gov
```

The first call downloads the embedding and rerank models (about 150 MB). The Docker image bakes them in.

## Project structure

```
carrag/
  corpus.py   corpus loading + rule-based make/model/year/source filter extraction
  store.py    Postgres schema, loading, keyword / dense search, local embedder + reranker
  rag.py      retrieve -> RRF -> rerank -> answer (Claude or mock) with citation validation
  app.py      FastAPI + static chat page
  seed.py     idempotent index build used by compose
eval/         make_questions.py (Claude-written eval set, run once), run_eval.py
scripts/      fetch_data.py: builds data/corpus.jsonl.gz from public APIs
data/         corpus.jsonl.gz + embeddings.npz (committed); raw/ API dumps are git-ignored
results/      retrieval.json, answers_mock.json
```

## Testing

```bash
MOCK_MODE=1 DATABASE_URL=... pytest -q     # 18 tests
ruff check .
```

The tests cover filter extraction (aliases, ambiguity, unknown vehicles), tag levels, corpus scrubbing (no VINs, emails or phone numbers), RRF, the retrieval metrics, mock refusal, and retrieval, refusal and the HTTP API against Postgres. CI runs them against a pgvector service container. The first run is slow (about 7 minutes locally) because it downloads and runs the models on CPU.

## What I learned / next steps

- **Filters beat model upgrades.** Rule-based vehicle filters moved hit@5 more than any retriever change, and an unknown-vehicle check refuses a whole class of questions without a model call.
- **AND-ed full-text queries return nothing for natural questions.** `plainto_tsquery` needs the `&` rewritten to `|`, then rank decides.
- **Mock refusal is only as good as its signal.** Refusing on "vehicle not indexed" catches the easy cases and misses off-topic questions, and nothing here measures how the live model refuses.
- **Next:** run the live eval once credit is available and add unanswerable questions back to measure refusals; add HNSW past ~100k rows; hand-write part of the eval set so it isn't all model-generated.

**One-page pitch:** [docs/PITCH.md](docs/PITCH.md) · **Design log:** [docs/DECISIONS.md](docs/DECISIONS.md)
