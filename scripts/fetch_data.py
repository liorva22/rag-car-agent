"""Build data/corpus.jsonl.gz from public-domain US government sources (no key needed).

    python scripts/fetch_data.py

- NHTSA recalls API      -> one doc per recall campaign, tagged with every covered vehicle-year
- NHTSA complaints API   -> a seeded sample of owner complaints per vehicle-year (VINs dropped, contact info scrubbed)
- FuelEconomy.gov CSV    -> one doc per EPA vehicle configuration
"""

import csv
import gzip
import io
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "corpus.jsonl.gz"
RAW = ROOT / "data" / "raw"
YEARS = range(2015, 2025)
COMPLAINTS_PER_VEHICLE = 20

# (make, NHTSA model name, FuelEconomy.gov baseModel)
VEHICLES = [
    ("TOYOTA", "CAMRY", "Camry"), ("TOYOTA", "COROLLA", "Corolla"), ("TOYOTA", "RAV4", "RAV4"),
    ("TOYOTA", "TACOMA", "Tacoma"), ("TOYOTA", "HIGHLANDER", "Highlander"), ("TOYOTA", "PRIUS", "Prius"),
    ("HONDA", "CIVIC", "Civic"), ("HONDA", "ACCORD", "Accord"), ("HONDA", "CR-V", "CR-V"), ("HONDA", "PILOT", "Pilot"),
    ("FORD", "F-150", "F150"), ("FORD", "ESCAPE", "Escape"), ("FORD", "EXPLORER", "Explorer"),
    ("FORD", "FOCUS", "Focus"), ("FORD", "FUSION", "Fusion"),
    ("CHEVROLET", "SILVERADO 1500", "Silverado"), ("CHEVROLET", "EQUINOX", "Equinox"),
    ("CHEVROLET", "MALIBU", "Malibu"), ("CHEVROLET", "BOLT EV", "Bolt EV"),
    ("NISSAN", "ALTIMA", "Altima"), ("NISSAN", "ROGUE", "Rogue"), ("NISSAN", "SENTRA", "Sentra"),
    ("HYUNDAI", "ELANTRA", "Elantra"), ("HYUNDAI", "SONATA", "Sonata"), ("HYUNDAI", "TUCSON", "Tucson"),
    ("KIA", "SORENTO", "Sorento"), ("KIA", "SOUL", "Soul"),
    ("JEEP", "GRAND CHEROKEE", "Grand Cherokee"), ("JEEP", "WRANGLER", "Wrangler"),
    ("SUBARU", "OUTBACK", "Outback"), ("SUBARU", "FORESTER", "Forester"),
    ("TESLA", "MODEL 3", "Model 3"), ("RAM", "1500", "1500"), ("MAZDA", "CX-5", "CX-5"),
]  # fmt: skip

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")
VIN = re.compile(r"\b(?=[A-HJ-NPR-Z0-9]*\d)(?=[A-HJ-NPR-Z0-9]*[A-Z])[A-HJ-NPR-Z0-9]{17}\b")


def get_json(url: str) -> dict:
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rag-car-agent/1.0 (portfolio project)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 400:  # NHTSA answers 400 (with an empty result list) for model-years that don't exist
                return {}
            if attempt == 5:
                raise
        except OSError:
            if attempt == 5:
                raise
        time.sleep(2 ** (attempt + 1))  # 403/429/5xx: back off


def api(kind: str, make: str, model: str, year: int) -> list[dict]:
    """Cached in data/raw/ so re-runs don't hammer the API."""
    cache = RAW / f"{kind}-{make}-{model}-{year}.json".replace(" ", "_")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    q = urllib.parse.urlencode({"make": make, "model": model, "modelYear": year})
    path = "recalls/recallsByVehicle" if kind == "recalls" else "complaints/complaintsByVehicle"
    rows = get_json(f"https://api.nhtsa.gov/{path}?{q}").get("results", [])
    cache.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def scrub(text: str) -> str:
    return VIN.sub("[VIN]", PHONE.sub("[PHONE]", EMAIL.sub("[EMAIL]", text))).strip()


def nhtsa_docs() -> list[dict]:
    jobs = [(mk, md, y) for mk, md, _ in VEHICLES for y in YEARS]
    with ThreadPoolExecutor(4) as ex:
        recalls = list(ex.map(lambda j: api("recalls", *j), jobs))
        complaints = list(ex.map(lambda j: api("complaints", *j), jobs))

    docs, by_campaign = [], {}
    for (mk, md, y), rows in zip(jobs, recalls, strict=True):
        for r in rows:
            cid = r["NHTSACampaignNumber"]
            d = by_campaign.setdefault(
                cid,
                {
                    "id": f"recall:{cid}",
                    "source": "recall",
                    "title": f"NHTSA recall {cid}: {r['Component']}",
                    "component": r["Component"],
                    "date": r.get("ReportReceivedDate", ""),
                    "text": f"Summary: {r['Summary']}\nConsequence: {r['Consequence']}\nRemedy: {r['Remedy']}",
                    "vehicles": [],
                    "url": f"https://www.nhtsa.gov/recalls?nhtsaId={cid}",
                },
            )
            if (v := f"{mk}|{md}|{y}") not in d["vehicles"]:
                d["vehicles"].append(v)
    docs += by_campaign.values()

    for (mk, md, y), rows in zip(jobs, complaints, strict=True):
        rows = [r for r in rows if len(r.get("summary") or "") > 80]
        for r in random.Random(f"{mk}{md}{y}").sample(rows, min(COMPLAINTS_PER_VEHICLE, len(rows))):
            flags = [f for f in ("crash", "fire") if r.get(f)]
            docs.append(
                {
                    "id": f"complaint:{r['odiNumber']}",
                    "source": "complaint",
                    "title": f"Owner complaint {r['odiNumber']}: {y} {mk} {md} - {r['components']}",
                    "component": r["components"],
                    "date": r.get("dateComplaintFiled", ""),
                    "text": scrub(r["summary"])[:2000] + (f"\nReported: {', '.join(flags)}" if flags else ""),
                    "vehicles": [f"{mk}|{md}|{y}"],
                    "url": f"https://www.nhtsa.gov/vehicle/{y}/{urllib.parse.quote(mk)}/{urllib.parse.quote(md)}",
                }
            )
    return docs


def fuel_docs() -> list[dict]:
    url = "https://www.fueleconomy.gov/feg/epadata/vehicles.csv.zip"
    with urllib.request.urlopen(url, timeout=120) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    rows = csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8", errors="replace"))
    wanted = {(mk.lower(), base.lower()): (mk, md) for mk, md, base in VEHICLES}
    docs = []
    for r in rows:
        key = (r["make"].lower(), r["baseModel"].lower())
        if key not in wanted or int(r["year"]) not in YEARS:
            continue
        mk, md = wanted[key]
        ev = r["atvType"] == "EV"
        engine = f"electric motor {r['evMotor']}" if ev else f"{r['displ']} L {r['cylinders']}-cylinder"
        text = (
            f"{r['year']} {r['make']} {r['model']} ({r['VClass']}). Engine: {engine}. Transmission: {r['trany']}. "
            f"Drive: {r['drive']}. Fuel: {r['fuelType']}{' / ' + r['atvType'] if r['atvType'] else ''}. "
            f"EPA fuel economy: {r['city08']} city / {r['highway08']} highway / {r['comb08']} combined "
            f"{'MPGe' if ev else 'MPG'}."
            + (f" EPA range: {r['range']} miles." if ev and r["range"] not in ("", "0") else "")
            + f" Estimated annual fuel cost: ${r['fuelCost08']}."
        )
        docs.append(
            {
                "id": f"fuel:{r['id']}",
                "source": "fuel",
                "title": f"EPA fuel economy: {r['year']} {r['make']} {r['model']} {r['trany']}",
                "component": "FUEL ECONOMY",
                "date": r["year"],
                "text": text,
                "vehicles": [f"{mk}|{md}|{r['year']}"],
                "url": f"https://www.fueleconomy.gov/feg/PowerSearch.do?action=noform&path=1&year1={r['year']}&year2={r['year']}&make={urllib.parse.quote(r['make'])}&baseModel={urllib.parse.quote(r['baseModel'])}&srchtyp=ymm",
            }
        )
    return docs


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    docs = nhtsa_docs() + fuel_docs()
    counts = {s: sum(d["source"] == s for d in docs) for s in ("recall", "complaint", "fuel")}
    OUT.parent.mkdir(exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        for d in sorted(docs, key=lambda d: d["id"]):
            f.write(json.dumps(d) + "\n")
    print(f"wrote {len(docs)} docs to {OUT.relative_to(ROOT)}: {counts}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
