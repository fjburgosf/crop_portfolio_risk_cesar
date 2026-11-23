"""Step 03: observed and optimised area allocations, evaluated out of sample.

For each specification, segment and evaluation year T, with an estimation window of the W previous years:
  - eligible activities: harvested area > 0 and production > 0 in every window year and in T, mean area >= 10 ha;
  - yield index idx = yield / window mean of the activity's yield, truncated at 3;
  - observed allocation s0 = harvested-area shares of the last window year;
  - target MOTAD: min (1/W) sum_t d_t  s.t. d_t >= 1 - sum_c s_c idx_tc, d_t >= 0, sum_c s_c = 1,
    (1 - delta) s0_c <= s_c <= (1 + delta) s0_c   (linear programme, HiGHS);
  - minimum variance: min s' S s with the same constraints (SLSQP), S = covariance of idx in the window;
  - evaluation in T: portfolio index R = sum_c s_c idx_Tc and shortfall max(0, 1 - R).

Outputs (results/): oos_results.csv, crop_shifts.csv, descriptives.json

Usage (from the repository root):  python src/03_portfolio.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "processed" / "panel.csv"
RES = ROOT / "results"
SEG_YEARS = {1: list(range(2007, 2018)), 2: list(range(2019, 2026))}
MAIN = {"W": 5, "delta": 0.20, "area": "harvested", "cap": 3.0, "drop_maize_seg2": False, "drop_constant": False}
SPECS = {"main": {},
         "delta_0.10": {"delta": 0.10}, "delta_0.30": {"delta": 0.30}, "delta_0.50": {"delta": 0.50},
         "window_4": {"W": 4}, "window_6": {"W": 6},
         "yield_per_sown": {"area": "sown"}, "no_truncation": {"cap": np.inf},
         "no_maize_seg2": {"drop_maize_seg2": True},
         # activities whose reported yield is constant over the window look riskless to the optimiser;
         # this specification excludes them
         "no_constant_yield": {"drop_constant": True}}
MIN_HA = 10.0


def motad(idx, s0, delta):
    T, n = idx.shape
    c = np.r_[np.zeros(n), np.full(T, 1.0 / T)]
    A_ub = np.hstack([-idx, -np.eye(T)])
    b_ub = -np.ones(T)
    A_eq = np.r_[np.ones(n), np.zeros(T)][None, :]
    bounds = [((1 - delta) * s, (1 + delta) * s) for s in s0] + [(0, None)] * T
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=bounds, method="highs")
    if r.status != 0:
        raise RuntimeError(r.message)
    return r.x[:n] / r.x[:n].sum()


def min_var(idx, s0, delta):
    S = np.cov(idx, rowvar=False, bias=True)
    bounds = [((1 - delta) * s, (1 + delta) * s) for s in s0]
    r = minimize(lambda s: s @ S @ s, s0, jac=lambda s: 2 * S @ s, bounds=bounds,
                 constraints=[{"type": "eq", "fun": lambda s: s.sum() - 1, "jac": lambda s: np.ones_like(s)}],
                 method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    if not r.success:
        raise RuntimeError(r.message)
    return r.x / r.x.sum()


def lpm(R):
    return float(np.mean(np.maximum(0.0, 1.0 - R)))


def windows(p, cfg):
    """Yield the estimation window and evaluation year data for every admissible T."""
    W = cfg["W"]
    for seg, years in SEG_YEARS.items():
        d = p[p.segment == seg]
        if cfg["drop_maize_seg2"] and seg == 2:
            d = d[d.crop != "MAIZ"]
        d = d[(d[cfg["area"]] > 0) & (d.harvested > 0) & (d.production > 0)].copy()
        d["y"] = d.production / d[cfg["area"]]
        for k in range(W, len(years)):
            win, T = years[k - W:k], years[k]
            sub = d[d.year.isin(win + [T])]
            Y = sub.pivot_table(index="year", columns="act", values="y")
            A = sub.pivot_table(index="year", columns="act", values="harvested")
            ok = Y.notna().all() & (A.loc[win].mean() >= MIN_HA)
            Y, A = Y.loc[:, ok], A.loc[:, ok]
            mu = Y.loc[win].mean()
            idx = (Y / mu).clip(upper=cfg["cap"])
            if cfg["drop_constant"]:
                keep = idx.loc[win].std(ddof=0) > 1e-9
                Y, A, idx = Y.loc[:, keep], A.loc[:, keep], idx.loc[:, keep]
            s0 = (A.loc[win[-1]] / A.loc[win[-1]].sum()).values
            flat = int((idx.loc[win].std(ddof=0) <= 1e-9).sum())
            yield seg, win, T, list(Y.columns), idx.loc[win].values, idx.loc[T].values, s0, flat


def run(p, name, over):
    cfg = {**MAIN, **over}
    rows, shifts = [], []
    for seg, win, T, acts, I, iT, s0, flat in windows(p, cfg):
        s_m = motad(I, s0, cfg["delta"])
        s_v = min_var(I, s0, cfg["delta"])
        Rin = {k: I @ s for k, s in (("obs", s0), ("motad", s_m), ("minvar", s_v))}
        rows.append({"spec": name, "segment": seg, "window": f"{win[0]}-{win[-1]}", "test_year": T, "n_act": len(acts),
                     "n_constant_yield": flat,
                     **{f"in_lpm_{k}": lpm(R) for k, R in Rin.items()},
                     **{f"in_sd_{k}": float(R.std()) for k, R in Rin.items()},
                     **{f"in_min_{k}": float(R.min()) for k, R in Rin.items()},
                     "R_obs": float(iT @ s0), "R_motad": float(iT @ s_m), "R_minvar": float(iT @ s_v),
                     "moved_share_motad": float(0.5 * np.abs(s_m - s0).sum()),
                     "moved_share_minvar": float(0.5 * np.abs(s_v - s0).sum())})
        if name == "main":
            for a, a0, am, av in zip(acts, s0, s_m, s_v):
                shifts.append({"segment": seg, "test_year": T, "act": a, "s_obs": a0, "s_motad": am, "s_minvar": av})
    out = pd.DataFrame(rows)
    for k in ("obs", "motad", "minvar"):
        out[f"short_{k}"] = np.maximum(0, 1 - out[f"R_{k}"])
    return out, pd.DataFrame(shifts)


def descriptives(p):
    """Dispersion and co-movement of the yield index over each whole segment (activities present every year)."""
    out = {}
    for seg, years in SEG_YEARS.items():
        d = p[(p.segment == seg) & (p.harvested > 0) & (p.production > 0)].copy()
        d["y"] = d.production / d.harvested
        Y = d.pivot_table(index="year", columns="act", values="y").reindex(years)
        A = d.pivot_table(index="year", columns="act", values="harvested").reindex(years)
        ok = Y.notna().all() & (A.mean() >= MIN_HA)
        Y, A = Y.loc[:, ok], A.loc[:, ok]
        idx = (Y / Y.mean()).clip(upper=3)
        cv = idx.std(ddof=0)
        crop = pd.Series([c.split("|")[1] for c in idx.columns], index=idx.columns)
        w = A.mean()
        by_crop = (pd.DataFrame({"crop": crop, "cv": cv, "w": w})
                   .groupby("crop").apply(lambda g: pd.Series({"cv_wmean": np.average(g.cv, weights=g.w),
                                                               "area_mean_ha": g.w.sum(), "n_act": len(g)}),
                                          include_groups=False)
                   .sort_values("area_mean_ha", ascending=False))
        C = idx.corr().values
        iu = np.triu_indices_from(C, 1)
        same_crop = np.equal.outer(crop.values, crop.values)[iu]
        R0 = (idx * (w / w.sum())).sum(axis=1)
        out[str(seg)] = {"years": f"{years[0]}-{years[-1]}", "n_act": int(ok.sum()), "area_mean_ha": float(w.sum()),
                         "mean_corr": float(np.nanmean(C[iu])), "mean_corr_same_crop": float(np.nanmean(C[iu][same_crop])),
                         "mean_corr_diff_crop": float(np.nanmean(C[iu][~same_crop])),
                         "cv_activity_wmean": float(np.average(cv, weights=w)),
                         "cv_portfolio": float(R0.std(ddof=0) / R0.mean()),
                         "by_crop": by_crop.round(4).reset_index().to_dict(orient="records")}
    return out


def main():
    p = pd.read_csv(PANEL, dtype={"mun": str})
    p["act"] = p.mun + "|" + p.crop
    res, shifts = [], None
    for name, over in SPECS.items():
        r, s = run(p, name, over)
        res.append(r)
        if name == "main":
            shifts = s
    res = pd.concat(res, ignore_index=True)
    res.to_csv(RES / "oos_results.csv", index=False)
    shifts["crop"] = shifts.act.str.split("|").str[1]
    shifts.to_csv(RES / "crop_shifts.csv", index=False)
    (RES / "descriptives.json").write_text(json.dumps(descriptives(p), ensure_ascii=False, indent=1), encoding="utf-8")
    m = res[res.spec == "main"]
    print(m[["segment", "test_year", "n_act", "in_lpm_obs", "in_lpm_motad", "R_obs", "R_motad", "R_minvar"]].round(4))
    summ = res.groupby("spec")[["in_lpm_obs", "in_lpm_motad", "short_obs", "short_motad", "short_minvar"]].mean()
    summ["wins_motad"] = res.assign(w=res.R_motad > res.R_obs).groupby("spec").w.sum()
    summ["n_test"] = res.groupby("spec").size()
    print(summ.round(4))


if __name__ == "__main__":
    main()
