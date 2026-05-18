"""
Grid Search pour optimiser la stratégie Value beta-neutre
Sauvegarde les résultats dans outputs/gs_value_results.csv

Usage:
    python gridsearch_value.py
    python gridsearch_value.py --max-combos 250
    python gridsearch_value.py --resume
"""

from __future__ import annotations
import json, time, itertools, argparse, sys, os, io, contextlib
from pathlib import Path
import numpy as np
import pandas as pd

# chemins
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent  # equity-basket/
sys.path.insert(0, str(PROJECT_DIR / "src" / "backtest"))

from signals import ValueSignalGenerator, get_rebalance_dates
from allocation import AllocationEngine, beta_neutralize_weights
from engine import run_multifactor_backtest
from pca_statarb import compute_rolling_beta


# Paramètres fixes
FIXED = {
    "start_date":        "2001-01-01",
    "end_date":          "2025-12-31",
    "initial_capital":   1_000_000,
    "rebalance_freq":    "ME",
    "use_low_vol":       True,
    "allocation_method": "equal_weight",
    "tc_bps":            5,
}

# Grille de paramètres à tester
GRID = {
    "n_buckets":        [3, 5, 7, 10],
    "long_short_style": [
        # (styles de sélection long/short)
        "extreme_only",         # long = [1], short = [n]
        "extreme_plus_one",     # long = [n-1, n], short = [1, 2]
    ],
    "vol_window":       [40, 60, 90, 126],
    "beta_lookback":    [126, 189, 252, 504],
}
# Total: ~128 combos, chaque combo ~75s -> 2.7h

BETA_MAX = 0.15  # Contrainte sur le beta moyen


def load_data():
    """Charge les data depuis data/raw/"""
    data_dir = PROJECT_DIR / "data" / "raw"
    end = pd.Timestamp(FIXED["end_date"])

    prices = pd.read_parquet(data_dir / "prices.parquet", engine="fastparquet")
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().loc[:end]

    pe_ratios = pd.read_parquet(data_dir / "pe_ratios.parquet", engine="fastparquet")
    pe_ratios.index = pd.to_datetime(pe_ratios.index)
    pe_ratios = pe_ratios.sort_index().loc[:end]

    rf_df = pd.read_parquet(data_dir / "risk_free.parquet", engine="fastparquet")
    risk_free = rf_df.iloc[:, 0]
    risk_free.index = pd.to_datetime(risk_free.index)
    risk_free = risk_free.sort_index()
    start = pd.Timestamp(FIXED["start_date"])
    risk_free = risk_free.loc[start:end]

    universe = pd.read_parquet(data_dir / "universe.parquet", engine="fastparquet")
    universe["date"] = pd.to_datetime(universe["date"])

    # Construction d'un benchmark équipondéré pour le calcul du beta
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

    return {
        "prices": prices, "pe_ratios": pe_ratios, "risk_free": risk_free,
        "universe": universe, "spx": spx,
    }


def score_run(track, spx, rf_mean):
    """Calcule les métriques de perf pour une combinaison"""
    rets = track["net_return"].values
    nd = len(rets); ny = nd / 252
    tr = track["net_value"].iloc[-1] / track["net_value"].iloc[0] - 1
    cagr = (1 + tr) ** (1 / ny) - 1 if ny > 0 else 0
    vol = np.std(rets) * np.sqrt(252)
    sharpe = (cagr - rf_mean) / vol if vol > 0 else 0
    dd = (track["net_value"] / track["net_value"].cummax() - 1).min()
    calmar = cagr / abs(dd) if dd != 0 else 0

    rb = compute_rolling_beta(track, spx, window=63)
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
    """Génère toutes les combos de la grille"""
    keys = sorted(k for k in grid if k != "long_short_style")
    vals = [grid[k] for k in keys]
    styles = grid["long_short_style"]

    combos = []
    for base in itertools.product(*vals):
        base_dict = dict(zip(keys, base))
        for style in styles:
            combo = {**base_dict, "long_short_style": style}
            combos.append(combo)
    return combos


def resolve_buckets(n_buckets, style):
    """Convertit le style en buckets longs/shorts concrets
    Logique: PE bas = bucket 1 (pas cher) -> LONG
             PE haut = bucket N (cher) -> SHORT
    """
    if style == "extreme_only":
        return [1], [n_buckets]
    elif style == "extreme_plus_one":
        lb = [1, 2] if n_buckets > 2 else [1]
        sb = [n_buckets - 1, n_buckets] if n_buckets > 2 else [n_buckets]
        return lb, sb
    else:
        return [1], [n_buckets]


def main():
    # Fix encoding pour Windows
    import sys as _sys
    if hasattr(_sys.stdout, 'reconfigure'):
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(_sys.stderr, 'reconfigure'):
        _sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-combos", type=int, default=0,
                    help="Nombre de combos à tester (0=all)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = PROJECT_DIR / "outputs"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "gs_value_results.csv"
    best_path = out_dir / "gs_value_best.json"

    combos = expand_grid(GRID)
    total = len(combos)

    if args.max_combos and args.max_combos < total:
        rng = np.random.default_rng(123)
        idx = rng.choice(total, size=args.max_combos, replace=False)
        combos = [combos[i] for i in sorted(idx)]

    # Reprise
    old_df = None
    done_keys = set()
    if args.resume and csv_path.exists():
        old_df = pd.read_csv(csv_path)
        done_keys = set(old_df["combo_key"].values)

    # Chargement des data
    t0 = time.time()
    data = load_data()

    rf_mean = data["risk_free"].mean() / 100

    # Dates de rebalancement
    rebal_dates = get_rebalance_dates(
        pd.Timestamp(FIXED["start_date"]),
        pd.Timestamp(FIXED["end_date"]),
        frequency=FIXED["rebalance_freq"],
    )

    # Exécution
    results = []
    n_combos = len(combos)
    t_start = time.time()

    for ci, combo in enumerate(combos):
        combo_key = json.dumps(combo, sort_keys=True)
        if combo_key in done_keys:
            continue

        n_buck = combo["n_buckets"]
        style = combo["long_short_style"]
        vol_win = combo["vol_window"]
        beta_lb = combo["beta_lookback"]
        tc = FIXED["tc_bps"]

        long_b, short_b = resolve_buckets(n_buck, style)

        elapsed_total = time.time() - t_start
        eta = (elapsed_total / max(ci, 1)) * (n_combos - ci) if ci > 0 else 0
        label = f"B={n_buck} {style} vw={vol_win} betaLB={beta_lb}"

        t1 = time.time()
        try:
            # Génération des signaux
            sig_gen = ValueSignalGenerator(
                n_buckets=n_buck,
                long_buckets=long_b,
                short_buckets=short_b,
            )
            signals = sig_gen.generate_signals(
                prices=data["prices"],
                pe_ratios=data["pe_ratios"],
                rebalance_dates=rebal_dates,
                universe=data["universe"],
                use_low_vol=FIXED["use_low_vol"],
                vol_window=vol_win,
            )

            # Allocation EW avec side composite si low_vol activé
            alloc_signal_type = "value"
            if FIXED["use_low_vol"] and "side_vol" in signals.columns:
                # Combiner value + vol en un côté composite
                def _composite_side(row):
                    sv = row.get("side_value", "NEUTRAL")
                    svol = row.get("side_vol", "NEUTRAL")
                    if sv == "LONG" and svol != "SHORT":
                        return "LONG"
                    if sv == "SHORT" and svol != "LONG":
                        return "SHORT"
                    return "NEUTRAL"
                signals["side_composite"] = signals.apply(_composite_side, axis=1)
                alloc_signal_type = "composite"

            engine = AllocationEngine(
                method="equal_weight",
                vol_window=vol_win,
                long_target=1.0,
                short_target=-1.0,
            )
            weights_raw = engine.generate_weights(
                signals=signals, prices=data["prices"],
                rebalance_dates=rebal_dates,
                signal_type=alloc_signal_type,
            )

            if weights_raw.empty:
                raise ValueError("No weights generated")

            # 3. Neutralisation beta (vs proxy EW S&P 500)
            weights = beta_neutralize_weights(
                weights_raw, data["prices"], data["spx"],
                lookback=beta_lb,
            )

            # Backtest
            track = run_multifactor_backtest(
                data["prices"], weights,
                start_date=FIXED["start_date"],
                end_date=FIXED["end_date"],
                initial_capital=FIXED["initial_capital"],
                tc_bps=tc,
            )

            # Évaluation
            sc = score_run(track, data["spx"], rf_mean)
            dt = time.time() - t1

            row = {**combo, **sc, "elapsed_s": dt,
                   "long_buckets": str(long_b), "short_buckets": str(short_b),
                   "tc_bps": tc, "combo_key": combo_key}
            results.append(row)

        except Exception as e:
            row = {**combo, "sharpe": np.nan, "error": str(e), 
                   "tc_bps": tc, "combo_key": combo_key}
            results.append(row)

        # Sauvegarde incrémentale
        if len(results) % 5 == 0:
            _save(results, csv_path, old_df)
            old_df = pd.read_csv(csv_path) if csv_path.exists() else None
            results = []

    # Sauvegarde finale
    if results:
        _save(results, csv_path, old_df)

    # Sélection de la meilleure combo
    final_df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    if len(final_df):
        valid = final_df[final_df["sharpe"].notna()].copy()
        feasible = valid[valid["beta_mean_abs"] <= BETA_MAX]
        if feasible.empty:
            feasible = valid.nsmallest(5, "beta_mean_abs")

        best = feasible.sort_values("sharpe", ascending=False).iloc[0]

        best_dict = {}
        for k, v in best.items():
            if k in ("combo_key", "error", "elapsed_s"):
                continue
            if hasattr(v, "item"):
                v = v.item()
            # Conversion des buckets en listes si besoin
            if k in ("long_buckets", "short_buckets") and isinstance(v, str):
                v = json.loads(v)
            best_dict[k] = v
        with open(best_path, "w") as f:
            json.dump(best_dict, f, indent=2, default=str)

    total_h = (time.time() - t_start) / 3600


def _save(results, csv_path, old_df):
    """Sauvegarde incrémentale"""
    df = pd.DataFrame(results)
    if old_df is not None and len(old_df):
        df = pd.concat([old_df, df], ignore_index=True)
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
