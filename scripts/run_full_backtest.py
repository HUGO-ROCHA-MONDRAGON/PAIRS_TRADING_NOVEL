from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading_novel.core import score_pair, select_baseline, select_matching
from pairs_trading_novel.metrics import performance_table
from pairs_trading_novel.paths import missing_raw_files, processed_data_dir, raw_data_dir


def load_inputs(raw: Path) -> dict[str, pd.DataFrame | pd.Series | pd.DatetimeIndex]:
    log_prices = pd.read_parquet(raw / "log_prices.parquet")
    daily_returns = pd.read_parquet(raw / "daily_returns.parquet")
    universe = pd.read_parquet(raw / "universe.parquet")
    benchmark_raw = pd.read_parquet(raw / "benchmark.parquet")
    risk_free_raw = pd.read_parquet(raw / "risk_free.parquet")
    rebalance_dates = pd.read_parquet(raw / "rebalance_dates.parquet")
    for df in [log_prices, daily_returns, benchmark_raw, risk_free_raw]:
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
    universe["date"] = pd.to_datetime(universe["date"])
    benchmark = benchmark_raw.iloc[:, 0].pct_change().rename("benchmark")
    risk_free = ((risk_free_raw.iloc[:, 0] / 100.0) / 252.0).rename("rf_daily")
    return {
        "log_prices": log_prices,
        "daily_returns": daily_returns,
        "universe": universe,
        "benchmark": benchmark,
        "risk_free": risk_free,
        "rebalance_dates": pd.to_datetime(rebalance_dates.iloc[:, 0].values),
    }


def eligible_tickers(log_prices: pd.DataFrame, universe: pd.DataFrame, rebalance_date: pd.Timestamp, lookback_days: int, min_coverage: float) -> list[str]:
    active = set(universe.loc[universe["date"] == rebalance_date, "ticker"])
    history = log_prices[log_prices.index <= rebalance_date].tail(lookback_days)
    cols = [col for col in history.columns if col in active]
    if not cols:
        return []
    coverage = history[cols].notna().mean()
    return coverage[coverage >= min_coverage].index.tolist()


def candidate_indices(returns: pd.DataFrame, tickers: list[str], threshold: float) -> list[tuple[int, int]]:
    n = len(tickers)
    if threshold <= 0.0:
        return list(itertools.combinations(range(n), 2))
    matrix = returns[tickers].fillna(0.0).to_numpy(float)
    centered = matrix - matrix.mean(axis=0)
    scale = np.sqrt((centered**2).sum(axis=0))
    scale[scale < 1e-12] = np.nan
    normalised = centered / scale
    corr = normalised.T @ normalised
    i, j = np.triu_indices(n, k=1)
    keep = np.abs(corr[i, j]) >= threshold
    return list(zip(i[keep].tolist(), j[keep].tolist()))


def build_edges(log_prices: pd.DataFrame, daily_returns: pd.DataFrame, rebalance_date: pd.Timestamp, tickers: list[str], lookback_days: int, min_obs: int, adf_threshold: float, corr_threshold: float) -> pd.DataFrame:
    columns = ["asset_a", "asset_b", "orientation", "alpha", "beta", "mu", "sigma", "adf_t", "weight"]
    if len(tickers) < 2:
        return pd.DataFrame(columns=columns)
    lp_hist = log_prices[log_prices.index <= rebalance_date].tail(lookback_days)
    ret_hist = daily_returns[daily_returns.index <= rebalance_date].tail(lookback_days)
    rows = []
    for i, j in candidate_indices(ret_hist, tickers, corr_threshold):
        a, b = tickers[i], tickers[j]
        scored = score_pair(lp_hist[a], lp_hist[b], min_obs=min_obs, adf_t_threshold=adf_threshold)
        if scored is not None:
            rows.append({"asset_a": a, "asset_b": b, **scored})
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True)


def simulate_pair(pair: pd.Series, trade_index: pd.DatetimeIndex, log_prices: pd.DataFrame, daily_returns: pd.DataFrame, entry_z: float, exit_z: float, stop_z: float, one_way_tc: float) -> tuple[pd.Series, pd.Series]:
    asset_a, asset_b = pair["asset_a"], pair["asset_b"]
    if asset_a not in log_prices.columns or asset_b not in log_prices.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    orientation = pair["orientation"]
    alpha, beta, mu, sigma = float(pair["alpha"]), float(pair["beta"]), float(pair["mu"]), float(pair["sigma"])
    prices = log_prices.reindex(trade_index)
    y_price = prices[asset_a] if orientation == "a_on_b" else prices[asset_b]
    x_price = prices[asset_b] if orientation == "a_on_b" else prices[asset_a]
    zscore = (y_price - (alpha + beta * x_price) - mu) / sigma
    position = np.zeros(len(zscore))
    current = 0.0
    for idx, value in enumerate(zscore.to_numpy(float)):
        if not np.isfinite(value):
            current = 0.0
        elif current == 0.0:
            if value >= entry_z:
                current = -1.0
            elif value <= -entry_z:
                current = 1.0
        elif abs(value) <= exit_z or abs(value) >= stop_z:
            current = 0.0
        position[idx] = current
    pos = pd.Series(position, index=trade_index)
    y_ret = daily_returns[asset_a if orientation == "a_on_b" else asset_b].reindex(trade_index).fillna(0.0)
    x_ret = daily_returns[asset_b if orientation == "a_on_b" else asset_a].reindex(trade_index).fillna(0.0)
    pnl = y_ret - beta * x_ret
    gross = pos.shift(1).fillna(0.0) * pnl
    net = gross - pos.diff().abs().fillna(pos.abs().iloc[0]) * one_way_tc
    return gross.rename("gross"), net.rename("net")


def simulate_portfolio(selected: pd.DataFrame, trade_index: pd.DatetimeIndex, log_prices: pd.DataFrame, daily_returns: pd.DataFrame, entry_z: float, exit_z: float, stop_z: float, one_way_tc: float) -> tuple[pd.Series, pd.Series]:
    if selected.empty or len(trade_index) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    gross_parts, net_parts = [], []
    for _, pair in selected.iterrows():
        gross, net = simulate_pair(pair, trade_index, log_prices, daily_returns, entry_z, exit_z, stop_z, one_way_tc)
        if not gross.empty:
            gross_parts.append(gross)
            net_parts.append(net)
    if not gross_parts:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return pd.concat(gross_parts, axis=1).mean(axis=1), pd.concat(net_parts, axis=1).mean(axis=1)


def concat_parts(parts: list[pd.Series]) -> pd.Series:
    if not parts:
        return pd.Series(dtype=float)
    series = pd.concat(parts).sort_index()
    return series[~series.index.duplicated(keep="first")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=raw_data_dir(ROOT))
    parser.add_argument("--out-dir", type=Path, default=processed_data_dir(ROOT))
    parser.add_argument("--lookback-days", type=int, default=504)
    parser.add_argument("--min-obs", type=int, default=252)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--adf-threshold", type=float, default=-2.50)
    parser.add_argument("--corr-threshold", type=float, default=0.50)
    parser.add_argument("--max-pairs", type=int, default=25)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--stop-z", type=float, default=4.0)
    parser.add_argument("--one-way-tc", type=float, default=0.001)
    args = parser.parse_args()
    missing = missing_raw_files(ROOT, full=True)
    if missing and args.raw_dir == raw_data_dir(ROOT):
        raise SystemExit("Missing required raw files: " + ", ".join(missing))
    data = load_inputs(args.raw_dir)
    log_prices = data["log_prices"]
    daily_returns = data["daily_returns"]
    universe = data["universe"]
    rebalance_dates = data["rebalance_dates"]
    benchmark = data["benchmark"]
    risk_free = data["risk_free"]
    gross_matching_parts, net_matching_parts, gross_baseline_parts, net_baseline_parts = [], [], [], []
    edge_parts, selected_matching_parts, selected_baseline_parts, diagnostics = [], [], [], []
    start = time.time()
    for idx, (rebalance_date, next_rebalance) in enumerate(zip(rebalance_dates[:-1], rebalance_dates[1:])):
        iter_start = time.time()
        tickers = eligible_tickers(log_prices, universe, rebalance_date, args.lookback_days, args.min_coverage)
        if len(tickers) < 20:
            diagnostics.append({"reb_date": rebalance_date, "status": "skip", "n_eligible": len(tickers)})
            continue
        edges = build_edges(log_prices, daily_returns, rebalance_date, tickers, args.lookback_days, args.min_obs, args.adf_threshold, args.corr_threshold)
        selected_matching = select_matching(edges, args.max_pairs)
        selected_baseline = select_baseline(edges, args.max_pairs)
        trade_index = pd.DatetimeIndex([date for date in log_prices.index if rebalance_date < date <= next_rebalance])
        gross_matching, net_matching = simulate_portfolio(selected_matching, trade_index, log_prices, daily_returns, args.entry_z, args.exit_z, args.stop_z, args.one_way_tc)
        gross_baseline, net_baseline = simulate_portfolio(selected_baseline, trade_index, log_prices, daily_returns, args.entry_z, args.exit_z, args.stop_z, args.one_way_tc)
        if not gross_matching.empty:
            gross_matching_parts.append(gross_matching)
            net_matching_parts.append(net_matching)
        if not gross_baseline.empty:
            gross_baseline_parts.append(gross_baseline)
            net_baseline_parts.append(net_baseline)
        if not edges.empty:
            tmp = edges.copy()
            tmp["reb_date"] = rebalance_date
            edge_parts.append(tmp)
        if not selected_matching.empty:
            tmp = selected_matching.copy()
            tmp["reb_date"] = rebalance_date
            selected_matching_parts.append(tmp)
        if not selected_baseline.empty:
            tmp = selected_baseline.copy()
            tmp["reb_date"] = rebalance_date
            selected_baseline_parts.append(tmp)
        diagnostics.append({"reb_date": rebalance_date, "status": "ok", "n_eligible": len(tickers), "n_edges": len(edges), "n_pairs_matching": len(selected_matching), "n_pairs_baseline": len(selected_baseline), "trade_days": len(trade_index), "iter_sec": round(time.time() - iter_start, 2)})
        print(f"{idx + 1:03d}/{len(rebalance_dates) - 1:03d} {rebalance_date.date()} eligible={len(tickers)} edges={len(edges)} matching={len(selected_matching)} baseline={len(selected_baseline)}")
    returns = pd.DataFrame({
        "gross_matching": concat_parts(gross_matching_parts),
        "net_matching": concat_parts(net_matching_parts),
        "gross_baseline": concat_parts(gross_baseline_parts),
        "net_baseline": concat_parts(net_baseline_parts),
        "benchmark": benchmark,
        "rf_daily": risk_free,
    })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(args.out_dir / "strategy_returns.parquet")
    pd.DataFrame(diagnostics).to_csv(args.out_dir / "diagnostics.csv", index=False)
    if edge_parts:
        pd.concat(edge_parts, ignore_index=True).to_parquet(args.out_dir / "candidate_edges.parquet", index=False)
    if selected_matching_parts:
        pd.concat(selected_matching_parts, ignore_index=True).to_parquet(args.out_dir / "selected_pairs_matching.parquet", index=False)
    if selected_baseline_parts:
        pd.concat(selected_baseline_parts, ignore_index=True).to_parquet(args.out_dir / "selected_pairs_baseline.parquet", index=False)
    table = performance_table(returns[["gross_matching", "net_matching", "gross_baseline", "net_baseline", "benchmark"]], risk_free)
    table.to_csv(args.out_dir / "table_performance.csv", index=False)
    print(table.to_string(index=False))
    print(f"Saved outputs to {args.out_dir}")
    print(f"Elapsed seconds: {time.time() - start:.1f}")


if __name__ == "__main__":
    main()
