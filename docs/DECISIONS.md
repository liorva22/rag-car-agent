# Design decisions

| # | Decision | Alternatives | Why |
|---|----------|--------------|-----|
| 1 | **Public US government data** (NHTSA recalls + complaints, EPA fuel economy), no key needed | Scraped forums or reviews; a synthetic corpus | Public domain, real, reproducible with `scripts/fetch_data.py`. Complaints are individual reports, so the prompt says they are unverified. VINs, emails and phone numbers are scrubbed, and a test checks it. |
| 2 | **Complaints are a seeded sample** (20 per vehicle-year) | All complaints | Keeps the corpus at 9,168 records, so embeddings can be committed and CI stays fast. *Trade-off:* a symptom's count in the index is not its real frequency. |
| 3 | **One Postgres table** with a generated `tsvector`, a `vector(384)` column and a GIN tag array | A separate vector DB; Elasticsearch | Keyword, dense and filter search in one SQL statement. Supabase-compatible. |
| 4 | **Exact vector scan, no HNSW** | HNSW / IVFFlat | At 9k rows it takes a few ms, and filtered HNSW can return fewer than k rows. Add HNSW with iterative scan past ~100k rows. |
| 5 | **Local embeddings: bge-small via fastembed (ONNX)** | An embeddings API; sentence-transformers | Anthropic has no embeddings endpoint, and torch adds ~2 GB. Vectors for the corpus are committed as float16, so a fresh clone doesn't re-embed. |
| 6 | **OR-ed full-text query** (`&` rewritten to `\|`) | `plainto_tsquery` as is | AND-ing every term returns nothing for most natural questions. Ranking with `ts_rank_cd` sorts out the matches. |
| 7 | **Reciprocal rank fusion** (k=60) of keyword and dense top 50 | Weighted score blend | Scores from the two searches aren't comparable. RRF needs no tuning. |
| 8 | **Cross-encoder rerank of the top 30** | No reranker; an LLM reranker | Lifts hit@1 from 0.77 to 0.86 (with filters). Costs about 1 s on CPU, so it is a switch (`mode`), and the table shows the cheaper options. |
| 9 | **Rule-based filters** on make / model / year / record type, with aliases | An LLM query planner | Free, deterministic, unit-tested. Vehicle found for 100% of eval questions, correct for 99.5%. Source hint applies only when exactly one hint matches. |
| 10 | **Refuse in code when the vehicle isn't indexed** | Always ask the model | No model call, no cost, no chance of an answer from general knowledge. |
| 11 | **`NOT_IN_SOURCES` sentinel** and a prompt that restricts answers to records for the asked vehicle | Free-text "I don't know" | Machine-checkable refusal, so the eval can measure it. |
| 12 | **Citations are validated** against the retrieved ids | Trust the model | A made-up id is reported in `invalid_citations` and not returned as a citation. |
| 13 | **Mock answerer is extractive** (first two sentences of the top record) | A canned string | Retrieval, citation plumbing and refusal stay real with no key. *Limitation:* answer correctness can't be measured in mock mode, and the README says so. |
| 14 | **`REFUSE_BELOW` defaults to 0** | Tune a score threshold on the eval set | Tuning on the same 230 questions that are reported would inflate the refusal number. Off-topic refusal in mock mode is reported as it is (20%). |
| 15 | **Eval set written by Claude from sampled gold records**, 200 answerable + 30 unanswerable, generated once and committed | Hand-written; real user queries | Scales and has a known gold record. *Limitation:* the generator saw the gold record, so questions can echo it. Stated in the README. |
| 16 | **Fuel questions count same-config trims as relevant** | Exact gold id only | Several EPA trims share identical MPG, so any of them answers the question. |
| 17 | **LLM judge is a different, cheaper model (Haiku) than the answerer (Sonnet)**, tolerant parsing of its verdict | Same model; strict JSON | The verdicts are read by regex because a stray quote in the notes field crashed two runs. |
| 18 | **Single shared DB connection behind a lock** in the API | A connection pool | Fine for a demo. Use `psycopg_pool` if concurrent users matter. |
