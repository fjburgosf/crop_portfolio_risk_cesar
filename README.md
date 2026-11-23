# Crop area allocation and productive risk in Cesar, Colombia (EVA 2007–2025): analysis code

This repository contains the analysis code used to evaluate, out of sample, whether reallocating the harvested area
among the municipality and crop activities of the department of Cesar (Colombia) reduces productive risk. It uses
the Municipal Agricultural Evaluations (Evaluaciones Agropecuarias Municipales, EVA). The observed allocation of the
harvested area is compared with a target MOTAD allocation and a minimum-variance allocation. Each estimation window
of W years is followed by one evaluation year.

The repository includes only the analysis scripts. The data are downloaded by step 01.

## Data

The EVA records are public and are downloaded from the open data portal of the Colombian government (datos.gov.co):

| Period | Source | Dataset |
|---|---|---|
| 2007–2018 | Ministerio de Agricultura y Desarrollo Rural | `2pnw-mmge` |
| 2019–2025 | Unidad de Planificación Rural Agropecuaria (UPRA) | `uejq-wxrr` |

Step 01 downloads the records of Cesar (department 20) and writes their URL, download date, number of rows and
SHA-256 in `data/raw/PROVENANCE.json`. The portal is updated over time, so a later download could differ from the
one used here; compare the SHA-256 values if you need the same input.

## Installation

Python 3.12 was used.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Running the analysis

Run the scripts from the repository root, in this order:

```bash
python src/01_get_sources.py
python src/02_build_panel.py
python src/03_portfolio.py
```

| Step | Script | What it does | Output |
|---|---|---|---|
| 01 | `01_get_sources.py` | Downloads the EVA records of Cesar, with provenance | `data/raw/eva_2007_2018_cesar.csv`, `eva_2019_2025_cesar.csv`, `PROVENANCE.json` |
| 02 | `02_build_panel.py` | Builds the annual panel of municipality and crop activities for the two periods | `data/processed/panel.csv`, `results/audit_panel.json` |
| 03 | `03_portfolio.py` | Yield indices, observed, target MOTAD and minimum-variance allocations, and their evaluation in sample and out of sample for every specification | `results/oos_results.csv`, `crop_shifts.csv`, `descriptives.json` |

Step 03 takes about one minute on a desktop computer.

## Definitions

- Activity: a crop in a municipality. Transitory crops are reported by semester (A + B are added); permanent and
  annual crops have one annual record. The disaggregations of a crop (for example, traditional and technified
  maize) are added.
- Periods: 2007–2017 (MADR; 2006 and 2018 are excluded because they are incomplete) and 2019–2025 (UPRA). No
  estimation or evaluation window mixes the two periods.
- Yield: production / harvested area (or sown area in one sensitivity analysis).
- Yield index: yield of each year divided by the mean yield of the activity in the estimation window, truncated
  at 3.
- Eligible activity: harvested area and production in every year of the window and in the evaluation year, with a
  mean harvested area of at least 10 ha.
- Observed allocation: harvested-area shares of the last year of the window.
- Target MOTAD: minimises the mean shortfall of the portfolio index below 1 in the window (linear programme, HiGHS),
  with shares that add up to 1 and change by at most δ of their observed value.
- Minimum variance: minimises the variance of the portfolio index with the same constraints (SLSQP).
- Evaluation: in the year after the window, the portfolio index R = Σ s·idx and its shortfall max(0, 1 − R).

## Specifications

`03_portfolio.py` runs the main specification (W = 5 years, δ = 0.20, harvested area, index truncated at 3) and
these variations: δ = 0.10, 0.30 and 0.50; windows of 4 and 6 years; yields per sown area; index without truncation;
the second period without maize; and the evaluation without the activities whose reported yield is constant in the
window. The `spec` column of `results/oos_results.csv` identifies each one.
