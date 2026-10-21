"""Corpus loading and rule-based metadata filters (make / model / year / record type) extracted from the question."""

import gzip
import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus.jsonl.gz"

# Spellings people use that the generic pattern below doesn't cover.
EXTRA_ALIASES = {"SILVERADO 1500": [r"SILVERADO"], "1500": [r"RAM(?:\s*1500)?"], "MODEL 3": [r"TESLA\s*3"], "BOLT EV": [r"BOLT"]}
SOURCE_HINTS = {
    "recall": r"\brecall",
    "fuel": r"\bmpge?\b|fuel economy|gas mileage|fuel cost|\brange\b|\befficien|miles per gallon",
    "complaint": r"complain|owners? (?:report|say|mention|describe)|anyone (?:report|else)|\bother owners\b",
}


@cache
def load_corpus(path: Path = CORPUS) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tags(doc: dict) -> list[str]:
    """Every level a filter can target: MAKE, MAKE|MODEL, MAKE|MODEL|YEAR."""
    out = set()
    for v in doc["vehicles"]:
        mk, md, y = v.split("|")
        out |= {mk, f"{mk}|{md}", v}
    return sorted(out)


def _model_pattern(model: str) -> str:
    # "CR-V" matches CRV / CR V / CR-V; "F-150" matches F150; "CX-5" matches CX5
    parts = [re.escape(p) for p in re.split(r"[\s-]+", model)]
    alts = [r"[\s-]?".join(parts), *EXTRA_ALIASES.get(model, [])]
    return r"\b(?:" + "|".join(alts) + r")\b"


@dataclass(frozen=True)
class Vocab:
    models: tuple[tuple[str, str, re.Pattern], ...]  # (make, model, pattern)
    makes: frozenset[str]
    years: frozenset[int]


@cache
def vocab() -> Vocab:
    pairs = {tuple(v.split("|")[:2]) for d in load_corpus() for v in d["vehicles"]}
    years = {int(v.split("|")[2]) for d in load_corpus() for v in d["vehicles"]}
    # longest names first so "GRAND CHEROKEE" wins over shorter overlaps
    models = tuple((mk, md, re.compile(_model_pattern(md), re.I)) for mk, md in sorted(pairs, key=lambda p: -len(p[1])))
    return Vocab(models, frozenset(mk for mk, _ in pairs), frozenset(years))


@dataclass
class Filters:
    tag: str | None = None  # most specific vehicle tag found, e.g. "HONDA|ACCORD|2018"
    source: str | None = None  # recall / complaint / fuel, only when exactly one hint matches
    year: int | None = None
    unknown_vehicle: bool = False  # a year or make we have no data for was named


MAKE_ALIASES = {"CHEVY": "CHEVROLET", "VW": "VOLKSWAGEN"}
OTHER_MAKES = r"\b(?:BMW|AUDI|MERCEDES|VOLKSWAGEN|VW|VOLVO|LEXUS|ACURA|PORSCHE|GMC|BUICK|CADILLAC|DODGE|CHRYSLER|LINCOLN|MITSUBISHI|INFINITI|GENESIS|MINI|JAGUAR|LAND ROVER|RIVIAN|LUCID|POLESTAR|FIAT)\b"


def extract_filters(question: str) -> Filters:
    v, q = vocab(), question.upper()
    f = Filters()
    if m := re.search(r"\b(19[89]\d|20[0-3]\d)\b", q):
        f.year = int(m.group(1))
    named_makes = {MAKE_ALIASES.get(w, w) for w in re.findall(r"[A-Z]+", q)} & v.makes
    hits = [(mk, md) for mk, md, pat in v.models if pat.search(q)]
    if named_makes:
        hits = [h for h in hits if h[0] in named_makes] or hits
    if hits:
        mk, md = hits[0]
        f.tag = f"{mk}|{md}" + (f"|{f.year}" if f.year else "")
    elif named_makes:
        f.tag = sorted(named_makes)[0]
    if (f.year and f.year not in v.years) or (not f.tag and re.search(OTHER_MAKES, q)):
        f.unknown_vehicle = True
    found = [s for s, pat in SOURCE_HINTS.items() if re.search(pat, question, re.I)]
    f.source = found[0] if len(found) == 1 else None
    return f
