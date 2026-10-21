"""Generate the eval set once with Claude (live). Output is committed: eval/questions.json.

    python eval/make_questions.py            # needs ANTHROPIC_API_KEY, ~$0.5

- 200 answerable questions, each written from one sampled record (its gold record), stratified by record type.
- 30 unanswerable questions: uncovered makes, covered vehicles but off-topic, covered models in uncovered years.
"""

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from carrag.corpus import load_corpus  # noqa: E402
from carrag.rag import Claude, cost  # noqa: E402

OUT = Path(__file__).resolve().parent / "questions.json"
PER_SOURCE = {"recall": 80, "complaint": 70, "fuel": 50}
MODEL = "claude-sonnet-5-5"

GEN = """You write evaluation questions for a car-owner assistant that answers from NHTSA recalls, NHTSA owner complaints \
and EPA fuel-economy listings.

Given one record and the owner's vehicle, write ONE question that owner might type, which this record answers. Then write \
the reference answer: one or two sentences with the key facts from the record.

- Name the vehicle (year, make, model) the way a person would ("my 2018 Accord", "2019 Honda CR-V").
- Ask about the substance (the defect, the symptom, the MPG), in your own words. Don't copy phrases from the record and \
don't mention record ids or campaign numbers.
- For complaints, ask whether other owners have reported the symptom. For fuel records, mention the engine, drive or \
transmission only if needed to pick this configuration.

Reply with JSON only: {"question": "...", "answer": "..."}"""

UNANSWERABLE = """Write 30 questions car owners might ask a recall / complaint / fuel-economy assistant that its index \
CANNOT answer. The index only covers model years 2015-2024 of: {models}.

- 10 about makes not in the list (e.g. BMW, Lexus, Volkswagen), any year.
- 10 about vehicles in the list but topics outside recalls, owner complaints and EPA fuel economy: maintenance \
schedules, prices, insurance, towing how-to, tire pressure, resale value.
- 10 about models in the list but model years 2005-2012.

Vary phrasing. Reply with JSON only: {{"questions": [{{"question": "...", "kind": "uncovered_make|off_topic|uncovered_year"}}]}}"""


def parse_json(text: str):
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    return json.loads(text)


def main() -> int:
    docs = load_corpus()
    rng = random.Random(42)
    picks = []
    for source, n in PER_SOURCE.items():
        pool = [d for d in docs if d["source"] == source and len(d["text"]) > 120]
        for d in rng.sample(pool, n):
            picks.append((d, rng.choice(d["vehicles"])))

    llm, spent = Claude(MODEL), [0.0]

    def gen(pick):
        d, vehicle = pick
        mk, md, y = vehicle.split("|")
        prompt = f"Vehicle: {y} {mk} {md}\n\n<record>\n{d['title']}\n{d['text'][:2500]}\n</record>"
        text, i, o = llm(GEN, prompt, max_tokens=600)
        spent[0] += cost(MODEL, i, o)
        q = parse_json(text)
        return {
            "question": q["question"],
            "reference": q["answer"],
            "gold": d["id"],
            "vehicle": vehicle,
            "source": d["source"],
            "answerable": True,
        }

    with ThreadPoolExecutor(8) as ex:
        answerable = list(ex.map(gen, picks))
    answerable = [{"id": f"a{n:03d}", **q} for n, q in enumerate(answerable)]

    models = sorted({" ".join(v.split("|")[:2]) for d in docs for v in d["vehicles"]})
    text, i, o = llm(UNANSWERABLE.format(models=", ".join(models)), "Write the 30 questions.", max_tokens=4000)
    spent[0] += cost(MODEL, i, o)
    unanswerable = [
        {"id": f"u{n:03d}", "question": q["question"], "kind": q["kind"], "gold": None, "answerable": False}
        for n, q in enumerate(parse_json(text)["questions"])
    ]
    OUT.write_text(
        json.dumps({"generator": MODEL, "cost_usd": round(spent[0], 4), "questions": answerable + unanswerable}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(answerable)} + {len(unanswerable)} questions, ${spent[0]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
