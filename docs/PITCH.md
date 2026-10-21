# RAG Car Agent: pitch

## The pitch
A car owner asks "is there a recall on my 2018 Accord?" or "do other owners see this shudder?". Public NHTSA and EPA data answers both, but nobody reads it, and a plain LLM answers from memory and invents campaign numbers. This agent retrieves from 9,168 real records (recalls, owner complaints, fuel economy), answers only from them with a citation per claim, and refuses when the index doesn't cover the question. Hybrid search with vehicle filters and a reranker puts a relevant record in the top 5 for 99% of 200 test questions, at 25 ms without the reranker.

## Problem and who it's for
Teams building question-answering over a private or public corpus who need answers they can audit: every claim points to a record, unverified owner complaints are labeled as such, and "I don't know" is a first-class outcome.

## How it works
See the README diagram:
1. Rules extract make, model, year and record type from the question. An unindexed vehicle is refused with no model call.
2. Postgres runs full-text and pgvector search over the same table, narrowed by the filters. Reciprocal rank fusion merges them and a local cross-encoder reranks the top 30.
3. Claude gets the top 6 records and must cite ids or reply `NOT_IN_SOURCES`. Citations are validated against what was retrieved.
4. `MOCK_MODE=1` swaps the answerer for an extractive stand-in, so everything except answer quality runs with no key.

## Stack and design choices
- **One Postgres table for keyword, vector and filter search.** One query path, no separate vector store. *Trade-off:* exact scan, so HNSW is needed past ~100k rows.
- **Rule-based vehicle filters instead of an LLM query planner.** Free, deterministic and testable: the right vehicle was found for 99.5% of questions. *Trade-off:* only the 34 indexed models and fixed aliases.
- **Local embeddings and reranker (ONNX, no torch).** No embedding API dependency and a small image. *Trade-off:* reranking costs about 1 s on CPU.
- **Refuse in code where possible.** Unknown vehicle means no model call and no cost.

## What I'd do differently
- **Run the live eval.** The API credit ran out before a live run finished, so there is no measured answer correctness, faithfulness or real refusal rate. That is the most important missing number.
- **A better eval set.** The 200 questions were written by Claude from the gold records, so they flatter retrieval. Part of the set should be hand-written or taken from real owner phrasing.
- **Measure refusals.** I dropped the 30 unanswerable questions I had generated: the mock can't refuse them meaningfully (it refuses only unknown vehicles) and the live run didn't happen. The system is built to refuse, but the repo has no measured refusal rate.
- **Latency.** Reranking is 1 s p50 on CPU. A smaller candidate set, a GPU or a hosted reranker would fix it.

## Interview talking points
- Why filters beat a bigger model here (keyword hit@5 0.65 to 0.965), and what each retrieval stage buys (the table in the README).
- Why `plainto_tsquery` returns nothing for natural questions and how OR-ing fixes it.
- How refusal is split between code (unknown vehicle) and the model (`NOT_IN_SOURCES`), and how citations are validated.
- Being upfront that the eval set is AI-written, the live run didn't happen and refusals aren't measured.
