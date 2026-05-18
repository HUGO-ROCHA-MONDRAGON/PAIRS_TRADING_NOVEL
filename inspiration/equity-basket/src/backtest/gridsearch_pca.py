"""
Grid Search pour optimiser les paramètres de la strat PCA Stat-Arb
Sauvegarde les résultats dans outputs/gs_pca_results.csv

Usage:
    python gridsearch_pca.py
    python gridsearch_pca.py --max-combos 200
    python gridsearch_pca.py --resume
"""

from __future__ import annotations
import json, time, itertools, argparse, sys, os, io, contextlib
from pathlib import Path
import numpy as np
import pandas as pd

# chemins
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent  # equity-basket/
sys.path.insert(0, str(PROJECT_DIR / "src" / "backtest"))

from pca_statarb import PCAStatArbStrategy, compute_rolling_beta


# Paramètres fixes
FIXED = {
    "start_date":            "2001-01-01",
    "end_date":              "2025-12-31",
    "initial_capital":       1_000_000,
    "standardise_vol":       True,
    "exit_zscore":           0.0,
    "max_gross":             4.0,
    "tc_bps":                5,
    "half_life_min":         1,
    "half_life_max":         30,
    "max_resid_vol_mult":    4.0,
    "regime_vol_window":     63,
    "regime_vol_cap":        2.0,
    "rebalance_freq":        "ME",
    "beta_lookback":         252,
}

# Grille de paramètres à tester
GRID = {
    "n_components":           [3, 5, 8, 10],
    "lookback":               [126, 189, 252, 504],
    "min_history":            [126, 252],
    "zscore_window":          [40, 60, 90],
    "entry_zscore":           [1.0, 1.25, 1.5, 2.0],
    "min_holding_days":       [21, 42, 63],
    "max_holding_days":       [90, 120, 180],
    "max_positions_per_side": [30, 50],
    "vol_target":             [0.08, 0.10, 0.15, 0.20],
}
# Total: ~13k combos. Avec --max-combos 300 -> ~8h

BETA_MAX = 0.15  # Contrainte sur le beta moyen


def load_data():
    """Charge les data depuis data/raw/"""
    data_dir = PROJECT_DIR / "data" / "raw"
    end = pd.Timestamp(FIXED["end_date"])

    prices = pd.read_parquet(data_dir / "prices.parquet", engine="fastparquet")
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().loc[:end]

    rf_df = pd.read_parquet(data_dir / "risk_free.parquet", engine="fastparquet")
    risk_free = rf_df.iloc[:, 0]
    risk_free.index = pd.to_datetime(risk_free.index)
    risk_free = risk_free.sort_index()
    start = pd.Timestamp(FIXED["start_date"])
    risk_free = risk_free.loc[start:end]

    universe = pd.read_parquet(data_dir / "universe.parquet", engine="fastparquet")
    universe["date"] = pd.to_datetime(universe["date"])

    # Construction d'un benchmark équipondéré du SPX
    univ_dates = np.sort(universe["date"].unique())
    univ_groups = universe.groupby("date")["ticker"].apply(set).to_dict()
    daily_ret = prices.pct_change(fill_method=None)
    idx_snap = np.searchsorted(univ_dates, prices.index, side="right") - 1

    ew_list, prev_snap, prev_tk = [], None, None
    for i, d in enumerate(prices.index):
        si = idx_snap[i]
        if si < 0:
            ew_list.append(daily_ret.iloc[i].mean())
        else:
            if si != prev_snap:
                prev_snap = si
                prev_tk = list(univ_groups[univ_dates[si]] & set(prices.columns))
            ew_list.append(daily_ret.loc[d, prev_tk].mean())
    spx = (1 + pd.Series(ew_list, index=prices.index).fillna(0)).cumprod() * 1000
    spx.name = "EW_MARKET"

    benchmark = spx

    return {
        "prices": prices, "risk_free": risk_free,
        "universe": universe, "spx": spx, "benchmark": benchmark,
    }


def score_run(track, benchmark, rf_mean):
    """Calcule les métriques de perf pour une combinaison"""
    rets = track["net_return"].values
    nd = len(rets); ny = nd / 252
    tr = track["net_value"].iloc[-1] / track["net_value"].iloc[0] - 1
    cagr = (1 + tr) ** (1 / ny) - 1 if ny > 0 else 0
    vol = np.std(rets) * np.sqrt(252)
    sharpe = (cagr - rf_mean) / vol if vol > 0 else 0
    dd = (track["net_value"] / track["net_value"].cummax() - 1).min()
    calmar = cagr / abs(dd) if dd != 0 else 0

    rb = compute_rolling_beta(track, benchmark, window=63)
    beta_mean = rb.abs().mean()

    excess = rets - rf_mean / 252
    down = excess[excess < 0]
    ds = np.std(down) * np.sqrt(252) if len(down) > 0 else 1e-8
    sortino = (np.mean(excess) * 252) / ds if ds > 0 else 0

    costs = track["cumulative_costs"].iloc[-1] if "cumulative_costs" in track.columns else 0

    return {
        "total_return": tr, "cagr": cagr, "vol": vol,
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "max_dd": dd, "beta_mean_abs": beta_mean,
        "avg_positions": track["n_positions"].mean(),
        "avg_gross": track["gross_exposure"].mean(),
        "total_costs": costs,
    }


def expand_grid(grid):
    """Génère toutes les combinaisons de la grille"""
    keys = sorted(grid.keys())
    vals = [grid[k] for k in keys]
    return [dict(zip(keys, c)) for c in itertools.product(*vals)]


def save_incremental(results, csv_path, old_df=None):
    """Sauvegarde incrémentale des résultats"""
    df = pd.DataFrame(results)
    if old_df is not None and len(old_df):
        df = pd.concat([old_df, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return df


def main():
    # Fix encoding pour Windows
    import sys as _sys
    if hasattr(_sys.stdout, 'reconfigure'):
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(_sys.stderr, 'reconfigure'):
        _sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-combos", type=int, default=300,
                    help="Nombre de combos à tester (0=all)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = PROJECT_DIR / "outputs"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "gs_pca_results.csv"
    best_path = out_dir / "gs_pca_best.json"

    combos = expand_grid(GRID)
    total = len(combos)

    if args.max_combos and args.max_combos < total:
        rng = np.random.default_rng(42)
        idx = rng.choice(total, size=args.max_combos, replace=False)
        combos = [combos[i] for i in sorted(idx)]

    # Reprise
    old_df = None
    done_keys = set()
    if args.resume and csv_path.exists():
        old_df = pd.read_csv(csv_path)
        done_keys = set(old_df["combo_key"].values)

    # Chargement
    t0 = time.time()
    data = load_data()

    rf_mean = data["risk_free"].mean() / 100

    # Exécution de la gridsearch
    results = []
    n_combos = len(combos)
    t_start = time.time()

    for ci, combo in enumerate(combos):
        combo_key = json.dumps(combo, sort_keys=True)
        if combo_key in done_keys:
            continue

        params = {**FIXED, **combo}
        label = " | ".join(f"{k}={v}" for k, v in sorted(combo.items()))
        elapsed_total = time.time() - t_start
        eta = (elapsed_total / max(ci, 1)) * (n_combos - ci) if ci > 0 else 0

        t1 = time.time()
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                strat = PCAStatArbStrategy(**params)
                res = strat.run(
                    prices=data["prices"],
                    universe=data["universe"],
                    benchmark=data["benchmark"],
                    start_date=FIXED["start_date"],
                    end_date=FIXED["end_date"],
                    risk_free=data["risk_free"],
                )
            track = res["track"]
            sc = score_run(track, data["spx"], rf_mean)
            dt = time.time() - t1

            row = {**combo, **sc, "elapsed_s": dt, "combo_key": combo_key}
            results.append(row)

        except Exception as e:
            row = {**combo, "sharpe": np.nan, "error": str(e), "combo_key": combo_key}
            results.append(row)

        # Sauvegarde incrémentale
        if len(results) % 5 == 0:
            old_df = save_incremental(results, csv_path, old_df)
            results = []

    # Sauvegarde finale
    if results:
        old_df = save_incremental(results, csv_path, old_df)

    # Sélection du meilleur
    if old_df is not None and len(old_df):
        valid = old_df[old_df["sharpe"].notna()].copy()
        feasible = valid[valid["beta_mean_abs"] <= BETA_MAX]
        if feasible.empty:
            feasible = valid.nsmallest(5, "beta_mean_abs")

        best = feasible.sort_values("sharpe", ascending=False).iloc[0]

        best_dict = {k: (v.item() if hasattr(v, "item") else v)
                     for k, v in best.items()
                     if k not in ("combo_key", "error", "elapsed_s")}
        with open(best_path, "w") as f:
            json.dump(best_dict, f, indent=2, default=str)

    total_h = (time.time() - t_start) / 3600


if __name__ == "__main__":
    main()
