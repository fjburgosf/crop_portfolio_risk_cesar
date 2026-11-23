"""Step 02: annual panel of municipality x crop activities for Cesar.

- Segment 1: EVA MADR 2007-2017 (2006B and 2018 excluded: incomplete agricultural years).
- Segment 2: EVA UPRA 2019-2025.
- Transitory crops: semester A + semester B; permanent and annual crops: annual record (no double records).
- Disaggregations of a crop (e.g. traditional and technified maize) are summed.

Output: data/processed/panel.csv  (segment, mun, municipio, crop, cycle, year, sown, harvested, production)
        results/audit_panel.json

Usage (from the repository root):  python src/02_build_panel.py
"""
import json
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
RES = ROOT / "results"


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.upper().split())


def load(fname, cols, segment, years):
    d = pd.read_csv(RAW / fname, dtype=str)
    d = d.rename(columns=cols)
    d = d[d.year.astype(int).between(*years)].copy()
    for c in ["sown", "harvested", "production"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["crop"] = d.crop.map(norm)
    d["cycle"] = d.cycle.map(norm).replace({"ANUAL": "PERMANENTE"})
    d["mun"] = d.mun.str.zfill(5)
    d["segment"] = segment
    return d


def main():
    OUT.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    a = load("eva_2007_2018_cesar.csv",
             {"c_d_mun": "mun", "municipio": "municipio", "cultivo": "crop", "ciclo_de_cultivo": "cycle", "a_o": "year",
              "rea_sembrada_ha": "sown", "rea_cosechada_ha": "harvested", "producci_n_t": "production",
              "estado_fisico_produccion": "state"}, 1, (2007, 2017))
    b = load("eva_2019_2025_cesar.csv",
             {"c_digo_dane_municipio": "mun", "municipio": "municipio", "cultivo": "crop", "ciclo_del_cultivo": "cycle",
              "a_o": "year", "rea_sembrada": "sown", "rea_cosechada": "harvested", "producci_n": "production",
              "estado_f_sico_del_cultivo": "state"}, 2, (2019, 2025))
    d = pd.concat([a, b], ignore_index=True)
    d["year"] = d.year.astype(int)
    d["municipio"] = d.municipio.map(norm)
    keys = ["segment", "mun", "crop", "year"]
    p = (d.groupby(keys).agg(municipio=("municipio", "first"), cycle=("cycle", "first"), sown=("sown", "sum"),
                             harvested=("harvested", "sum"), production=("production", "sum"),
                             n_records=("crop", "size"), states=("state", lambda s: "|".join(sorted(set(s)))))
         .reset_index())
    p.to_csv(OUT / "panel.csv", index=False)
    audit = {"rows_raw": {"segment1": int(len(a)), "segment2": int(len(b))},
             "activities": {int(s): int(g[["mun", "crop"]].drop_duplicates().shape[0]) for s, g in p.groupby("segment")},
             "crops": {int(s): int(g.crop.nunique()) for s, g in p.groupby("segment")},
             "municipalities": {int(s): int(g.mun.nunique()) for s, g in p.groupby("segment")},
             "years": {int(s): sorted(map(int, g.year.unique())) for s, g in p.groupby("segment")},
             "harvested_ha_by_year": {int(y): round(float(v), 1) for y, v in p.groupby("year").harvested.sum().items()},
             "missing_values": {c: int(p[c].isna().sum()) for c in ["sown", "harvested", "production"]},
             "harvested_gt_sown": int((p.harvested > p.sown * 1.001).sum()),
             "physical_state_changes": {f"{s}-{c}": sorted(set(g.states)) for (s, c), g in p.groupby(["segment", "crop"])
                                        if g.states.nunique() > 1}}
    (RES / "audit_panel.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "harvested_ha_by_year"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
