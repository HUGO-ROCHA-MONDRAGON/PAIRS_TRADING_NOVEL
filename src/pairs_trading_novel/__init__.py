"""pairs_trading_novel — réplication OOP du papier Qureshi & Zaman (2024)."""
from .core import adf1_tstat, ols_beta_resid, score_pair, select_baseline, select_matching
from .metrics import (
    annual_return,
    annual_volatility,
    max_drawdown,
    performance_table,
    sharpe_ratio,
    sharpe_ratio_mean_std,
)
from .paths import find_project_root, raw_data_dir, processed_data_dir
from .data import DataLoader, MarketData
from .cointegration import PairAnalyzer, PairResult
from .selection import BaselineSelector, MatchingSelector, PairSelector
from .signals import (
    QScoreSignal,
    SpreadSpec,
    TradingSignal,
    ZScoreSignal,
    edge_to_spec,
    spread_series,
)
from .backtest import Backtester, BacktestConfig, BacktestResult
from .portfolio_metrics import (
    concentration_series,
    correlation_matrix,
    cumulative_return,
    jaccard_retention,
    sortino_ratio,
    summary_table,
    turnover_series,
)
from .theoretical import TheoremCalculator, TheoreticalParams
from . import visualisation

__all__ = [
    "adf1_tstat", "ols_beta_resid", "score_pair",
    "select_baseline", "select_matching",
    "annual_return", "annual_volatility", "max_drawdown",
    "sharpe_ratio", "sharpe_ratio_mean_std", "performance_table",
    "find_project_root", "raw_data_dir", "processed_data_dir",
    "DataLoader", "MarketData",
    "PairAnalyzer", "PairResult",
    "BaselineSelector", "MatchingSelector", "PairSelector",
    "QScoreSignal", "ZScoreSignal", "TradingSignal",
    "SpreadSpec", "edge_to_spec", "spread_series",
    "Backtester", "BacktestConfig", "BacktestResult",
    "concentration_series", "correlation_matrix", "cumulative_return",
    "jaccard_retention", "sortino_ratio", "summary_table", "turnover_series",
    "TheoremCalculator", "TheoreticalParams",
    "visualisation",
]
