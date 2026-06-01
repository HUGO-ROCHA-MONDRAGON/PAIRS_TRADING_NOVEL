"""
backtest.py — Boucle principale (POO) du backtest pairs‑trading.

Reproduit la procédure du papier :
    - rebalancement mensuel (77 dates 2017‑01 → 2023‑05)
    - fenêtre de formation glissante 504 jours (~2 ans)
    - sélecteur configurable (Baseline ou Matching)
    - signal configurable (ZScore ou QScore)
    - allocation $‑neutre β‑ajustée : pnl_pair = pos · (r_y − β·r_x)
    - coûts de transaction sur changement de position (1 voie = ``one_way_tc``)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from .cointegration import PairAnalyzer
from .data import MarketData
from .selection import PairSelector
from .signals import QScoreSignal, SpreadSpec, TradingSignal, ZScoreSignal, edge_to_spec, spread_series


@dataclass
class BacktestConfig:
    lookback_days: int = 504
    min_obs: int = 252
    min_coverage: float = 0.90
    adf_t_threshold: float | None = None
    adf_pvalue_threshold: float | None = None
    correlation_filter: float = 0.50
    max_candidate_pairs: int | None = None
    max_pairs: int = 25
    one_way_tc: float = 0.001
    annual_tc: float = 0.01
    tc_model: str = "paper_daily"  # "paper_daily" | "trade_change"
    min_eligible: int = 20
    start_date: pd.Timestamp | None = None
    end_date: pd.Timestamp | None = None
    progress: bool = True
    verbose: bool = True


@dataclass
class BacktestResult:
    gross: pd.Series
    net: pd.Series
    diagnostics: pd.DataFrame
    selected_pairs: pd.DataFrame
    edges: pd.DataFrame

    def cumulative(self, kind: str = "net") -> pd.Series:
        r = self.net if kind == "net" else self.gross
        return (1.0 + r.fillna(0.0)).cumprod() - 1.0


def _eligible_tickers(market: MarketData, date: pd.Timestamp,
                      lookback_days: int, min_coverage: float) -> list[str]:
    active = set(market.universe.loc[market.universe["date"] == date, "ticker"])
    hist = market.log_prices[market.log_prices.index <= date].tail(lookback_days)
    cols = [c for c in hist.columns if c in active]
    if not cols:
        return []
    coverage = hist[cols].notna().mean()
    return coverage[coverage >= min_coverage].index.tolist()


def _formation_quantiles(spec_row: pd.Series, market: MarketData,
                         date: pd.Timestamp, lookback_days: int) -> tuple[float, float, float]:
    """Estime τ25/τ50/τ75 du spread sur la fenêtre de formation."""
    if spec_row["orientation"] == "a_on_b":
        ya, xa = spec_row["asset_a"], spec_row["asset_b"]
    else:
        ya, xa = spec_row["asset_b"], spec_row["asset_a"]
    lp = market.log_prices[market.log_prices.index <= date].tail(lookback_days)
    spread = lp[ya] - (float(spec_row["alpha"]) + float(spec_row["beta"]) * lp[xa])
    spread = spread.dropna()
    if spread.empty:
        return float("nan"), float("nan"), float("nan")
    return (
        float(spread.quantile(0.25)),
        float(spread.quantile(0.50)),
        float(spread.quantile(0.75)),
    )


class Backtester:
    """
    Orchestrateur. Usage :
        bt = Backtester(market, selector=MatchingSelector(25),
                        signal=ZScoreSignal(2.0))
        result = bt.run()
    """

    def __init__(
        self,
        market: MarketData,
        selector: PairSelector,
        signal: TradingSignal,
        analyzer: PairAnalyzer | None = None,
        config: BacktestConfig | None = None,
    ) -> None:
        self.market = market
        self.selector = selector
        self.signal = signal
        self.config = config or BacktestConfig()
        self.analyzer = analyzer or PairAnalyzer(
            min_obs=self.config.min_obs,
            adf_t_threshold=self.config.adf_t_threshold,
            adf_pvalue_threshold=self.config.adf_pvalue_threshold,
            correlation_filter=self.config.correlation_filter,
            max_candidate_pairs=self.config.max_candidate_pairs,
        )

    # ------------------------------------------------------------------
    def _simulate_pair(
        self, pair: pd.Series, trade_idx: pd.DatetimeIndex
    ) -> tuple[pd.Series, pd.Series]:
        market = self.market
        cfg = self.config
        a, b = pair["asset_a"], pair["asset_b"]
        if a not in market.log_prices.columns or b not in market.log_prices.columns:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        # Construction de la spec (q‑score nécessite les quantiles)
        if isinstance(self.signal, QScoreSignal):
            tau25, tau50, tau75 = _formation_quantiles(
                pair, market, pair.get("reb_date", trade_idx[0]), cfg.lookback_days
            )
            spec = SpreadSpec(
                asset_y=a if pair["orientation"] == "a_on_b" else b,
                asset_x=b if pair["orientation"] == "a_on_b" else a,
                alpha=float(pair["alpha"]), beta=float(pair["beta"]),
                mu=float(pair["mu"]), sigma=float(pair["sigma"]),
                tau25=tau25, tau50=tau50, tau75=tau75,
            )
        else:
            spec = edge_to_spec(pair)

        pos = self.signal.positions(spec, market.log_prices, trade_idx)
        # P&L journalier β‑ajusté
        r_y = market.daily_returns[spec.asset_y].reindex(trade_idx).fillna(0.0)
        r_x = market.daily_returns[spec.asset_x].reindex(trade_idx).fillna(0.0)
        raw_pnl = r_y - spec.beta * r_x
        held_pos = pos.shift(1).fillna(0.0)
        gross = held_pos * raw_pnl

        if cfg.tc_model == "paper_daily":
            # Modèle papier: 1% annuel appliqué en coût quotidien.
            # Portfolio pair long/short -> deux jambes => facteur 2.
            daily_tc = cfg.annual_tc / 252.0
            net = gross - held_pos.abs() * (2.0 * daily_tc)
        elif cfg.tc_model == "trade_change":
            # Legacy: coût uniquement lors des changements de position.
            trades = pos.diff().abs().fillna(pos.abs().iloc[0] if len(pos) else 0.0)
            net = gross - trades * cfg.one_way_tc
        else:
            raise ValueError(f"Unknown tc_model={cfg.tc_model}")
        return gross.rename("gross"), net.rename("net")

    # ------------------------------------------------------------------
    def _simulate_portfolio(
        self, selected: pd.DataFrame, trade_idx: pd.DatetimeIndex
    ) -> tuple[pd.Series, pd.Series]:
        if selected.empty or len(trade_idx) == 0:
            empty = pd.Series(dtype=float)
            return empty, empty
        g_parts, n_parts = [], []
        for _, row in selected.iterrows():
            g, n = self._simulate_pair(row, trade_idx)
            if not g.empty:
                g_parts.append(g)
                n_parts.append(n)
        if not g_parts:
            empty = pd.Series(dtype=float)
            return empty, empty
        return (
            pd.concat(g_parts, axis=1).mean(axis=1),
            pd.concat(n_parts, axis=1).mean(axis=1),
        )

    # ------------------------------------------------------------------
    def run(self, rebalance_dates: Iterable[pd.Timestamp] | None = None) -> BacktestResult:
        market = self.market
        cfg = self.config
        dates = list(rebalance_dates) if rebalance_dates is not None else list(market.rebalance_dates)
        if cfg.start_date is not None:
            start_date = pd.Timestamp(cfg.start_date)
            dates = [date for date in dates if date >= start_date]
        if cfg.end_date is not None:
            end_date = pd.Timestamp(cfg.end_date)
            dates = [date for date in dates if date <= end_date]
        gross_parts, net_parts = [], []
        all_edges_parts, sel_parts, diag = [], [], []
        t0 = time.time()
        windows = list(zip(dates[:-1], dates[1:]))
        n = len(windows)
        iterator = enumerate(windows)
        if cfg.progress:
            iterator = enumerate(
                tqdm(windows, total=n, desc="Rebalances", unit="reb")
            )
        for idx, (reb, nxt) in iterator:
            t_iter = time.time()
            tickers = _eligible_tickers(market, reb, cfg.lookback_days, cfg.min_coverage)
            if len(tickers) < cfg.min_eligible:
                diag.append({"reb_date": reb, "status": "skip", "n_eligible": len(tickers)})
                continue
            edges = self.analyzer.build_edges(
                market.log_prices, market.daily_returns, tickers, reb, cfg.lookback_days
            )
            selected = self.selector(edges)
            if not selected.empty:
                selected = selected.copy()
                selected["reb_date"] = reb
            trade_idx = pd.DatetimeIndex(
                [d for d in market.log_prices.index if reb < d <= nxt]
            )
            g, nrt = self._simulate_portfolio(selected, trade_idx)
            if not g.empty:
                gross_parts.append(g)
                net_parts.append(nrt)
            if not edges.empty:
                e = edges.copy(); e["reb_date"] = reb; all_edges_parts.append(e)
            if not selected.empty:
                sel_parts.append(selected)
            diag.append({
                "reb_date": reb, "status": "ok", "n_eligible": len(tickers),
                "n_edges": len(edges), "n_pairs": len(selected),
                "trade_days": len(trade_idx), "iter_sec": round(time.time() - t_iter, 2),
            })
            if cfg.verbose:
                print(f"[{idx+1:3d}/{n}] {reb.date()} elig={len(tickers):4d} "
                      f"edges={len(edges):5d} sel={len(selected):3d} "
                      f"days={len(trade_idx):3d} {time.time()-t_iter:5.1f}s")
        gross = _concat_series(gross_parts)
        net = _concat_series(net_parts)
        edges_df = pd.concat(all_edges_parts, ignore_index=True) if all_edges_parts else pd.DataFrame()
        sel_df = pd.concat(sel_parts, ignore_index=True) if sel_parts else pd.DataFrame()
        diag_df = pd.DataFrame(diag)
        if cfg.verbose:
            print(f"Backtest done in {time.time()-t0:.1f}s — {len(net)} days")
        return BacktestResult(
            gross=gross, net=net, diagnostics=diag_df,
            selected_pairs=sel_df, edges=edges_df,
        )


def _concat_series(parts: list[pd.Series]) -> pd.Series:
    if not parts:
        return pd.Series(dtype=float)
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated(keep="first")]
