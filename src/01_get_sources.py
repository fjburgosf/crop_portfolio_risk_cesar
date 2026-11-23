"""Step 01: download the Municipal Agricultural Evaluations (EVA) for Cesar from datos.gov.co, with provenance.

- EVA 2007-2018, Ministerio de Agricultura y Desarrollo Rural (dataset 2pnw-mmge)
- EVA 2019-2025, UPRA, Base Agrícola (dataset uejq-wxrr)

Output: data/raw/eva_2007_2018_cesar.csv, eva_2019_2025_cesar.csv, PROVENANCE.json (URL, date, rows, SHA-256)

Usage (from the repository root):  python src/01_get_sources.py
"""
import datetime as dt
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SETS = {"eva_2007_2018_cesar.csv": ("2pnw-mmge", "c_d_dep"),
        "eva_2019_2025_cesar.csv": ("uejq-wxrr", "c_digo_dane_departamento")}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    prov = {}
    for fname, (ds, depcol) in SETS.items():
        url = f"https://www.datos.gov.co/resource/{ds}.csv"
        params = {"$where": f"{depcol} in ('20', '020')", "$limit": 200000}
        r = requests.get(url, params=params, timeout=300)
        r.raise_for_status()
        (RAW / fname).write_bytes(r.content)
        rows = r.content.decode("utf-8").count("\n") - 1
        prov[fname] = {"dataset": ds, "url": r.url, "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "rows": rows, "sha256": hashlib.sha256(r.content).hexdigest(),
                       "licence": "CC BY-SA 4.0 (datos.gov.co)"}
        print(fname, rows)
    (RAW / "PROVENANCE.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
