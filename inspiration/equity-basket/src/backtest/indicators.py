"""
Indicateurs de performance et métriques de risque pour les backtests.
Classe PerformanceAnalyzer + fonctions autonomes pour calculer Sharpe, Sortino, VaR, CVaR, DD, etc.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
from scipy import stats


class PerformanceAnalyzer:
    """
    Classe qui regroupe toutes les métriques de perf (Sharpe, Sortino, DD, VaR, etc).
    Pratique pour éviter de répéter les mêmes arguments track/rf/benchmark à chaque calcul.
    """

    def __init__(
        self,
        track: pd.DataFrame,
        risk_free_rate: Union[float, pd.Series] = 0.0,
        benchmark_returns: Optional[pd.Series] = None,
        return_col: str = "net_return",
        value_col: str = "net_value",
    ):
        self.track = track
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns = benchmark_returns
        self.return_col = return_col
        self.value_col = value_col

    def returns(self) -> Dict[str, float]:
        return calculate_returns(self.track, self.return_col, self.value_col)

    def volatility(self, annualize: bool = True) -> float:
        return calculate_volatility(self.track, self.return_col, annualize)

    def sharpe(self, annualize: bool = True) -> float:
        return calculate_sharpe_ratio(self.track, self.risk_free_rate, self.return_col, annualize)

    def sortino(self, annualize: bool = True) -> float:
        return calculate_sortino_ratio(self.track, self.risk_free_rate, self.return_col, annualize)

    def calmar(self) -> float:
        return calculate_calmar_ratio(self.track, self.return_col, self.value_col)

    def drawdown_series(self) -> pd.Series:
        return calculate_drawdown(self.track, self.value_col)

    def max_drawdown(self) -> Dict[str, float]:
        return calculate_max_drawdown(self.track, self.value_col)

    def win_rate(self) -> Dict[str, float]:
        return calculate_win_rate(self.track, self.return_col)

    def profit_factor(self) -> float:
        return calculate_profit_factor(self.track, self.return_col)

    def summary(self) -> Dict[str, object]:
        return track_summary(self.track, self.return_col, self.value_col)

    def tracking_error(self, annualize: bool = True) -> float:
        if self.benchmark_returns is None:
            return 0.0
        return calculate_tracking_error(self.track, self.benchmark_returns, self.return_col, annualize)

    def information_ratio(self, annualize: bool = True) -> float:
        if self.benchmark_returns is None:
            return 0.0
        return calculate_information_ratio(self.track, self.benchmark_returns, self.return_col, annualize)

    def beta(self) -> float:
        if self.benchmark_returns is None:
            return 0.0
        return calculate_beta(self.track, self.benchmark_returns, self.return_col)

    def all_metrics(self) -> Dict[str, float]:
        """Calcule toutes les métriques d'un coup."""
        return calculate_all_metrics(
            self.track, self.risk_free_rate, self.benchmark_returns,
            self.return_col, self.value_col,
        )

    def format_table(self) -> pd.DataFrame:
        """Formate les métriques en DataFrame pour l'affichage."""
        return format_metrics_table(self.all_metrics())


# Fonctions autonomes pour calculs de métriques

def calculate_returns(
    track: pd.DataFrame,
    return_col: str = 'net_return',
    value_col: str = 'net_value'
) -> Dict[str, float]:
    """Retourne total, annualisé, moyenne et écart-type quotidiens."""
    returns = track[return_col].dropna()
    # Calcul à partir de la valeur finale ou des rendements composés
    if value_col in track.columns:
        total_return = (track[value_col].iloc[-1] / track[value_col].iloc[0]) - 1
        total_value  = track[value_col].iloc[-1]
    else:
        total_return = (1 + returns).prod() - 1
        total_value  = (1 + returns).prod()
    n_days = len(returns)
    years = n_days / 252  # jours de trading par an
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    daily_mean = returns.mean()
    daily_std = returns.std()
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'daily_mean_return': daily_mean,
        'daily_std_return': daily_std,
        'total_value': total_value,
        'n_days': n_days,
        'n_years': years
    }


def calculate_volatility(
    track: pd.DataFrame,
    return_col: str = 'net_return',
    annualize: bool = True
) -> float:
    """Volatilité journalière ou annualisée (sqrt(252))."""
    returns = track[return_col].dropna()
    vol = returns.std()
    if annualize:
        vol = vol * np.sqrt(252)  # annualisation
    return vol


def calculate_sharpe_ratio(
    track: pd.DataFrame,
    risk_free_rate: Union[float, pd.Series] = 0.0,
    return_col: str = 'net_return',
    annualize: bool = True
) -> float:
    """Ratio de Sharpe. Rendements excédentaires / écart-type."""
    returns = track[return_col].dropna()
    # Conversion du taux rf en taux journalier
    if isinstance(risk_free_rate, (int, float)):
        rf_daily = risk_free_rate / 252
        excess_returns = returns - rf_daily
    else:
        rf_aligned = risk_free_rate.reindex(returns.index, method='ffill')
        rf_daily = rf_aligned / 252 / 100
        excess_returns = returns - rf_daily
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()
    if std_excess == 0:
        return 0.0
    sharpe = mean_excess / std_excess
    if annualize:
        sharpe = sharpe * np.sqrt(252)
    return sharpe


def calculate_sortino_ratio(
    track: pd.DataFrame,
    risk_free_rate: Union[float, pd.Series] = 0.0,
    return_col: str = 'net_return',
    annualize: bool = True
) -> float:
    """Ratio de Sortino. Comme Sharpe mais avec la downside volatility."""
    returns = track[return_col].dropna()
    # Rf journalier
    if isinstance(risk_free_rate, (int, float)):
        rf_daily = risk_free_rate / 252
        excess_returns = returns - rf_daily
    else:
        rf_aligned = risk_free_rate.reindex(returns.index, method='ffill')
        rf_daily = rf_aligned / 252 / 100
        excess_returns = returns - rf_daily
    mean_excess = excess_returns.mean()
    downside_returns = excess_returns[excess_returns < 0]  # seulement les pertes
    if len(downside_returns) == 0:
        return np.inf
    downside_std = downside_returns.std()
    if downside_std == 0:
        return 0.0
    sortino = mean_excess / downside_std
    if annualize:
        sortino = sortino * np.sqrt(252)
    return sortino


def calculate_calmar_ratio(
    track: pd.DataFrame,
    return_col: str = 'net_return',
    value_col: str = 'net_value'
) -> float:
    """Ratio de Calmar. CAGR / Max DD."""
    ret = calculate_returns(track, return_col, value_col)
    cagr = ret['annualized_return']
    dd = calculate_max_drawdown(track, value_col)
    max_dd = abs(dd['max_drawdown'])
    if max_dd == 0:
        return np.inf
    return cagr / max_dd


def calculate_drawdown(track: pd.DataFrame, value_col: str = 'net_value') -> pd.Series:
    """Serie de drawdown jour par jour."""
    values = track[value_col]
    running_max = values.expanding().max()
    drawdown = (values - running_max) / running_max
    return drawdown


def calculate_max_drawdown(track: pd.DataFrame, value_col: str = 'net_value') -> Dict[str, float]:
    """Max drawdown avec dates de pic et creux."""
    drawdown = calculate_drawdown(track, value_col)
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    # On remonte pour trouver le pic avant le max DD
    drawdown_before_max = drawdown[:max_dd_idx]
    peak_idx = (drawdown_before_max == 0).last_valid_index()
    if peak_idx is None:
        peak_idx = drawdown.index[0]
    duration = len(track[peak_idx:max_dd_idx])
    peak_date = track.loc[peak_idx, 'date'] if 'date' in track.columns else peak_idx
    trough_date = track.loc[max_dd_idx, 'date'] if 'date' in track.columns else max_dd_idx
    return {
        'max_drawdown': max_dd,
        'max_drawdown_pct': max_dd * 100,
        'drawdown_duration_days': duration,
        'peak_date': peak_date,
        'trough_date': trough_date,
        'peak_value': track.loc[peak_idx, value_col],
        'trough_value': track.loc[max_dd_idx, value_col]
    }


def calculate_var(
    track: pd.DataFrame,
    return_col: str = 'net_return',
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> float:
    """Value at Risk. Méthode historique ou paramétrique."""
    returns = track[return_col].dropna()
    if method == 'historical':
        var = -returns.quantile(1 - confidence_level)
    elif method == 'parametric':
        mean = returns.mean()
        std = returns.std()
        z_score = stats.norm.ppf(1 - confidence_level)
        var = -(mean + z_score * std)
    else:
        raise ValueError(f"Méthode inconnue: {method}")
    return var


def calculate_cvar(
    track: pd.DataFrame,
    return_col: str = 'net_return',
    confidence_level: float = 0.95
) -> float:
    """CVaR (Expected Shortfall). Moyenne des pertes pires que la VaR."""
    returns = track[return_col].dropna()
    var = -returns.quantile(1 - confidence_level)
    # Rendements dans la queue de distribution
    tail_returns = returns[returns <= -var]
    if len(tail_returns) == 0:
        return var
    cvar = -tail_returns.mean()
    return cvar


def calculate_win_rate(
    track: pd.DataFrame,
    return_col: str = 'net_return'
) -> Dict[str, float]:
    """Win rate + stats sur jours gagnants vs perdants."""
    returns = track[return_col].dropna()
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    n_win = len(wins)
    n_loss = len(losses)
    total = n_win + n_loss
    return {
        'win_rate':        n_win / total if total > 0 else 0.0,
        'n_winning_days':  n_win,
        'n_losing_days':   n_loss,
        'avg_win':         wins.mean()  if n_win  > 0 else 0.0,
        'avg_loss':        losses.mean() if n_loss > 0 else 0.0,
        'best_day':        returns.max(),
        'worst_day':       returns.min(),
    }


def calculate_profit_factor(
    track: pd.DataFrame,
    return_col: str = 'net_return'
) -> float:
    """Profit factor. Somme gains / somme pertes."""
    returns = track[return_col].dropna()
    gross_profit = returns[returns > 0].sum()
    gross_loss   = returns[returns < 0].sum()
    if gross_loss == 0:
        return np.inf
    return gross_profit / abs(gross_loss)


def track_summary(
    track: pd.DataFrame,
    return_col: str = 'net_return',
    value_col: str = 'net_value'
) -> Dict[str, object]:
    """Résumé du track: dates, valeurs, coûts, turnover, expositions."""
    summary: Dict[str, object] = {}
    summary['start_date'] = track['date'].iloc[0] if 'date' in track.columns else track.index[0]
    summary['end_date']   = track['date'].iloc[-1] if 'date' in track.columns else track.index[-1]
    summary['n_days']     = len(track)
    if value_col in track.columns:
        summary['initial_value'] = track[value_col].iloc[0]
        summary['final_value']   = track[value_col].iloc[-1]
        summary['total_pnl']     = summary['final_value'] - summary['initial_value']
    if 'transaction_cost' in track.columns:
        summary['total_transaction_costs'] = track['transaction_cost'].sum()
        summary['avg_daily_cost']          = track['transaction_cost'].mean()
    elif 'cumulative_costs' in track.columns:
        summary['total_transaction_costs'] = track['cumulative_costs'].iloc[-1]
    if 'turnover' in track.columns:
        summary['avg_turnover']   = track[track['turnover'] > 0]['turnover'].mean()
        summary['total_turnover'] = track['turnover'].sum()
    if 'n_positions' in track.columns:
        summary['avg_n_positions'] = track['n_positions'].mean()
    if 'gross_exposure' in track.columns:
        summary['avg_gross_exposure'] = track['gross_exposure'].mean()
    if 'net_exposure' in track.columns:
        summary['avg_net_exposure'] = track['net_exposure'].mean()
    return summary


def calculate_tracking_error(
    track: pd.DataFrame,
    benchmark_returns: pd.Series,
    return_col: str = 'net_return',
    annualize: bool = True
) -> float:
    """Tracking Error. Écart-type des rendements actifs (portfolio - benchmark)."""
    portfolio_returns = track[return_col]
    if 'date' in track.columns:
        portfolio_returns = track.set_index('date')[return_col]
    active_returns = portfolio_returns - benchmark_returns
    active_returns = active_returns.dropna()
    if len(active_returns) == 0:
        return 0.0
    te = active_returns.std()
    if annualize:
        te = te * np.sqrt(252)
    return te


def calculate_information_ratio(
    track: pd.DataFrame,
    benchmark_returns: pd.Series,
    return_col: str = 'net_return',
    annualize: bool = True
) -> float:
    """Information Ratio. Surperf moyenne / tracking error."""
    portfolio_returns = track[return_col]
    if 'date' in track.columns:
        portfolio_returns = track.set_index('date')[return_col]
    active_returns = portfolio_returns - benchmark_returns
    active_returns = active_returns.dropna()
    if len(active_returns) == 0:
        return 0.0
    mean_active = active_returns.mean()
    std_active = active_returns.std()
    if std_active == 0:
        return 0.0
    ir = mean_active / std_active
    if annualize:
        ir = ir * np.sqrt(252)
    return ir


def calculate_beta(
    track: pd.DataFrame,
    benchmark_returns: pd.Series,
    return_col: str = 'net_return'
) -> float:
    """Beta du portfolio vs benchmark. Cov(port, bench) / Var(bench)."""
    portfolio_returns = track[return_col]
    if 'date' in track.columns:
        portfolio_returns = track.set_index('date')[return_col]
    aligned = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()
    if len(aligned) < 2:
        return 0.0
    cov = aligned['portfolio'].cov(aligned['benchmark'])
    var_bm = aligned['benchmark'].var()
    if var_bm == 0:
        return 0.0
    return cov / var_bm


def calculate_all_metrics(
    track: pd.DataFrame,
    risk_free_rate: Union[float, pd.Series] = 0.0,
    benchmark_returns: Optional[pd.Series] = None,
    return_col: str = 'net_return',
    value_col: str = 'net_value'
) -> Dict[str, float]:
    """Calcule toutes les métriques disponibles en un seul appel."""
    metrics = {}
    # Rendements et vol
    return_metrics = calculate_returns(track, return_col, value_col)
    metrics.update(return_metrics)
    metrics['volatility_annual'] = calculate_volatility(track, return_col, annualize=True)
    metrics['volatility_daily']  = calculate_volatility(track, return_col, annualize=False)
    # Ratios
    metrics['sharpe_ratio']  = calculate_sharpe_ratio(track, risk_free_rate, return_col)
    metrics['sortino_ratio'] = calculate_sortino_ratio(track, risk_free_rate, return_col)
    metrics['calmar_ratio']  = calculate_calmar_ratio(track, return_col, value_col)
    # DD
    dd_metrics = calculate_max_drawdown(track, value_col)
    metrics.update(dd_metrics)
    # VaR et CVaR
    metrics['var_95_hist'] = calculate_var(track, return_col, 0.95, method='historical')
    metrics['var_99_hist'] = calculate_var(track, return_col, 0.99, method='historical')
    metrics['cvar_95']     = calculate_cvar(track, return_col, 0.95)
    metrics['cvar_99']     = calculate_cvar(track, return_col, 0.99)
    metrics['es_95']       = metrics['cvar_95']   # ES = CVaR
    metrics['es_99']       = metrics['cvar_99']
    # Win rate et profit factor
    wr = calculate_win_rate(track, return_col)
    metrics.update(wr)
    metrics['profit_factor'] = calculate_profit_factor(track, return_col)
    # Résumé du track
    ts = track_summary(track, return_col, value_col)
    for k, v in ts.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            metrics[k] = v
    rf_annual = float(risk_free_rate) if isinstance(risk_free_rate, (int, float)) else 0.0
    cagr = metrics.get('annualized_return', 0.0)
    metrics['alpha_vs_rf'] = cagr - rf_annual
    # Métriques vs benchmark si dispo
    if benchmark_returns is not None:
        metrics['tracking_error']    = calculate_tracking_error(track, benchmark_returns, return_col)
        metrics['information_ratio'] = calculate_information_ratio(track, benchmark_returns, return_col)
        metrics['beta']              = calculate_beta(track, benchmark_returns, return_col)
    return metrics


def format_metrics_table(metrics: Dict[str, float]) -> pd.DataFrame:
    """Formate les métriques en DataFrame avec catégories et valeurs formatées."""
    categories = {
        'Rendement': [
            ('total_return',        'Total Return',        'pct'),
            ('annualized_return',   'CAGR',                'pct'),
            ('total_value',         'Total Value (NAV)',   'money'),
            ('daily_mean_return',   'Daily Mean Return',   'bp'),
        ],
        'Risque': [
            ('volatility_annual',   'Annualised Vol',      'pct'),
            ('max_drawdown',        'Max Drawdown',        'pct'),
        ],
        'VaR / CVaR': [
            ('var_95_hist',  'VaR 95 % (hist)',   'pct'),
            ('var_99_hist',  'VaR 99 % (hist)',   'pct'),
            ('cvar_95',      'CVaR 95 %',         'pct'),
            ('cvar_99',      'CVaR 99 %',         'pct'),
        ],
        'Ratios': [
            ('sharpe_ratio',  'Sharpe Ratio',  'dec3'),
            ('sortino_ratio', 'Sortino Ratio', 'dec3'),
            ('calmar_ratio',  'Calmar Ratio',  'dec3'),
            ('alpha_vs_rf',   'Alpha vs Rf (ann.)', 'pct'),
        ],
        'Trading': [
            ('win_rate',               'Win Rate',              'pct'),
            ('profit_factor',          'Profit Factor',         'dec2'),
            ('avg_n_positions',        'Avg Positions',         'dec0'),
            ('avg_gross_exposure',     'Avg Gross Exposure',    'dec2'),
            ('avg_net_exposure',       'Avg Net Exposure',      'dec4'),
            ('total_transaction_costs','Total Costs',           'money'),
            ('avg_turnover',           'Avg Turnover (rebal)',  'pct'),
        ],
        'Benchmark': [
            ('beta',              'Beta',              'dec3'),
        ],
    }
    rows = []
    for category, specs in categories.items():
        for key, label, fmt in specs:
            if key not in metrics:
                continue
            value = metrics[key]
            if fmt == 'pct':
                formatted = f"{value * 100:.2f}%"
            elif fmt == 'bp':
                formatted = f"{value * 10_000:.2f} bp"
            elif fmt == 'dec0':
                formatted = f"{value:.0f}"
            elif fmt == 'dec2':
                formatted = f"{value:.2f}"
            elif fmt == 'dec3':
                formatted = f"{value:.3f}"
            elif fmt == 'dec4':
                formatted = f"{value:.4f}"
            elif fmt == 'money':
                formatted = f"${value:,.0f}"
            elif fmt == 'int':
                formatted = f"{int(value)}"
            else:
                formatted = f"{value}"
            rows.append({'Category': category, 'Metric': label, 'Value': formatted})
    return pd.DataFrame(rows)
