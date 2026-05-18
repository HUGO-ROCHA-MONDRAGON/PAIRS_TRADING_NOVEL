"""
Graphiques de backtest : courbes d'équité, tableaux de bord de risque,
comparaisons multi-stratégies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')

COLORS = ['#2ecc71', '#3498db', '#e67e22', '#9b59b6', '#e74c3c',
          '#1abc9c', '#f39c12']


class BacktestVisualizer:
    """Classe pour tracer les résultats de backtest."""

    def __init__(
        self,
        tracks: dict[str, pd.DataFrame],
        benchmark: pd.Series | None = None,
        rf_level: pd.Series | None = None,
        spx: pd.Series | None = None,
        rolling_betas: dict[str, pd.Series] | None = None,
    ):
        self.tracks = tracks
        self.benchmark = benchmark
        self.rf_level = rf_level
        self.spx = spx
        self.rolling_betas = rolling_betas or {}

    def plot_equity_curve(self, name: str | None = None,
                          title: str | None = None) -> None:
        """Courbe d'équité + drawdown pour une stratégie."""
        if name is None:
            name = next(iter(self.tracks))
        track = self.tracks[name]
        rf = self.rf_level if self.rf_level is not None else pd.Series(dtype=float)
        spx = self.spx if self.spx is not None else pd.Series(dtype=float)
        plot_equity_curve(track, rf, spx, title=title)

    def plot_risk_dashboard(self, name: str | None = None) -> None:
        """Tableau de bord risque pour une stratégie."""
        if name is None:
            name = next(iter(self.tracks))
        rb = self.rolling_betas.get(name, pd.Series(dtype=float))
        plot_risk_dashboard(self.tracks[name], rb)

    def plot_multi_equity(self, title: str = "Equity Curves (base 100)") -> None:
        """Courbes multi-stratégies."""
        plot_multi_equity(self.tracks, self.benchmark, title)

    def plot_multi_risk_dashboard(self) -> None:
        """Tableau de bord multi-stratégies."""
        plot_multi_risk_dashboard(self.tracks, self.rolling_betas)

    @staticmethod
    def plot_metrics_comparison(metrics_dict: dict) -> pd.DataFrame:
        """Tableau comparatif des métriques."""
        return plot_metrics_comparison(metrics_dict)

    def print_beta_diagnostics(self, name: str | None = None) -> None:
        """Stats du beta glissant."""
        if name is None:
            name = next(iter(self.rolling_betas))
        print_beta_diagnostics(self.rolling_betas[name])


def plot_equity_curve(track: pd.DataFrame,
                      rf_level: pd.Series,
                      spx: pd.Series,
                      title: str = None) -> None:
    """Courbe d'équité base 100 + drawdown."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10),
                             gridspec_kw={'height_ratios': [3, 1]})

    dates = track['date']
    base_val = track['net_value'].iloc[0]

    ax = axes[0]  # équité
    ax.plot(dates, track['net_value'] / base_val * 100,
            color='#2ecc71', linewidth=2,
            label='Strategie')

    # Rf
    bench_aligned = rf_level.reindex(dates.values, method='ffill').values
    bench_mask = ~np.isnan(bench_aligned)
    if bench_mask.any():
        first_valid = bench_mask.argmax()
        bench_base = bench_aligned[first_valid]
        if bench_base > 0:
            bench_norm = bench_aligned / bench_base * 100
            ax.plot(dates[bench_mask], bench_norm[bench_mask],
                    color='#e74c3c', linewidth=1.5, linestyle='--',
                    label='Treasury Bill 3M')
    else:
        bench_norm = np.full(len(dates), np.nan)

    # EW
    spx_aligned = spx.reindex(dates.values, method='ffill').values
    spx_mask = ~np.isnan(spx_aligned)
    if spx_mask.any():
        first_valid_spx = spx_mask.argmax()
        spx_base = spx_aligned[first_valid_spx]
        if spx_base > 0:
            spx_norm = spx_aligned / spx_base * 100
            ax.plot(dates[spx_mask], spx_norm[spx_mask],
                    color='#3498db', linewidth=1.5, linestyle='-.',
                    alpha=0.8, label='Marche EW')
    else:
        spx_norm = np.full(len(dates), np.nan)

    ax.axhline(100, color='gray', linewidth=0.5, linestyle=':')
    ax.set_title(title or 'Equity Curve (Base 100)',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Value')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]  # drawdown
    cummax = track['net_value'].cummax()
    dd = (track['net_value'] / cummax - 1) * 100
    ax2.fill_between(dates, dd, 0, alpha=0.4, color='#e74c3c')
    ax2.plot(dates, dd, color='#e74c3c', linewidth=0.8)
    ax2.set_title('Drawdown (%)', fontsize=12)
    ax2.set_ylabel('Drawdown %')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_risk_dashboard(pca_track: pd.DataFrame,
                        rolling_beta: pd.Series) -> None:
    """4 panneaux : beta, exposition, positions, turnover."""
    fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)

    dates = pca_track['date']

    axes[0].plot(dates, rolling_beta.values, color='#3498db', linewidth=1)
    axes[0].axhline(0, color='red', linewidth=0.8, linestyle='--')
    axes[0].fill_between(dates, rolling_beta.values, 0,
                         alpha=0.15, color='#3498db')
    axes[0].set_title('Beta glissant (63j)',
                      fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Beta')
    axes[0].set_ylim(-0.5, 0.5)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dates, pca_track['gross_exposure'],
                 color='#2ecc71', linewidth=1, label='Gross')
    axes[1].plot(dates, pca_track['net_exposure'],
                 color='#e74c3c', linewidth=1, label='Net')
    axes[1].axhline(0, color='gray', linewidth=0.5, linestyle=':')
    axes[1].set_title('Gross & Net Exposure', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Exposure')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(dates, pca_track['n_positions'],
                 color='#9b59b6', linewidth=1)
    axes[2].set_title('Nombre de positions',
                      fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Count')
    axes[2].grid(True, alpha=0.3)

    rebal_mask = pca_track['is_rebalance']
    axes[3].bar(dates[rebal_mask],
                pca_track.loc[rebal_mask, 'turnover'],
                width=3, color='#e67e22', alpha=0.7)
    axes[3].set_title('Turnover (jours rebalancement)',
                      fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Turnover')
    axes[3].set_xlabel('Date')
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def print_beta_diagnostics(rolling_beta: pd.Series) -> None:
    """Stats du beta glissant."""
    rb = rolling_beta.dropna()
    print(f"Beta glissant ({len(rb)} jours):")
    print(f"  Moy: {rb.mean():.4f}")
    print(f"  Med: {rb.median():.4f}")
    print(f"  Std: {rb.std():.4f}")
    print(f"  Min: {rb.min():.4f}")
    print(f"  Max: {rb.max():.4f}")


def plot_multi_equity(tracks: dict,
                      benchmark: pd.Series = None,
                      title: str = "Equity Curves (base 100)") -> None:
    """Courbes d'équité multi-stratégies."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10),
                             gridspec_kw={'height_ratios': [3, 1]})

    ax = axes[0]
    for i, (name, track) in enumerate(tracks.items()):
        dates = track['date'] if 'date' in track.columns else track.index
        base = track['net_value'].iloc[0]
        eq = track['net_value'] / base * 100
        c = COLORS[i % len(COLORS)]
        ax.plot(dates, eq, color=c, linewidth=2, label=name)
        ax.annotate(f'{eq.iloc[-1]:.0f}', xy=(dates.iloc[-1], eq.iloc[-1]),
                    fontsize=9, fontweight='bold', color=c,
                    xytext=(10, 5 - i*15), textcoords='offset points')

    if benchmark is not None:
        first_track = list(tracks.values())[0]
        d0 = first_track['date'].iloc[0] if 'date' in first_track.columns else first_track.index[0]
        de = first_track['date'].iloc[-1] if 'date' in first_track.columns else first_track.index[-1]
        ba = benchmark.loc[d0:de]
        if len(ba) > 0 and ba.notna().any():
            ba_clean = ba.dropna()
            ax.plot(ba_clean.index, ba_clean / ba_clean.iloc[0] * 100,
                    color='gray', linewidth=1.2, linestyle='--',
                    alpha=0.7, label='Treasury Bill 3M')

    ax.axhline(100, color='gray', linewidth=0.5, linestyle=':')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Value')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for i, (name, track) in enumerate(tracks.items()):
        dates = track['date'] if 'date' in track.columns else track.index
        dd = (track['net_value'] / track['net_value'].cummax() - 1) * 100
        c = COLORS[i % len(COLORS)]
        ax2.plot(dates, dd, color=c, linewidth=0.8, label=name, alpha=0.8)
    ax2.set_title('Drawdown (%)', fontsize=12)
    ax2.set_ylabel('DD %')
    ax2.set_xlabel('Date')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_multi_risk_dashboard(tracks: dict,
                              rolling_betas: dict = None) -> None:
    """Tableau de bord multi-stratégies."""
    fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)

    for i, (name, track) in enumerate(tracks.items()):
        dates = track['date'] if 'date' in track.columns else track.index
        c = COLORS[i % len(COLORS)]

        if rolling_betas and name in rolling_betas:
            rb = rolling_betas[name]
            axes[0].plot(dates, rb.reindex(dates.values if hasattr(dates, 'values') else dates).values,
                         color=c, linewidth=1, label=name, alpha=0.8)

        if 'gross_exposure' in track.columns:
            axes[1].plot(dates, track['gross_exposure'], color=c, linewidth=1,
                         label=f'{name} Gross', alpha=0.8)
            axes[1].plot(dates, track['net_exposure'], color=c, linewidth=1,
                         linestyle='--', alpha=0.5)

        if 'n_positions' in track.columns:
            axes[2].plot(dates, track['n_positions'], color=c, linewidth=1,
                         label=name, alpha=0.8)

        if 'turnover' in track.columns:
            rebal = track[track['is_rebalance']] if 'is_rebalance' in track.columns else track
            rd = rebal['date'] if 'date' in rebal.columns else rebal.index
            axes[3].bar(rd, rebal['turnover'], width=3, color=c,
                        alpha=0.5, label=name)

    axes[0].axhline(0, color='red', linewidth=0.8, linestyle='--')
    axes[0].set_title('Rolling Beta vs Benchmark', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Beta')
    axes[0].set_ylim(-0.5, 0.5)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(0, color='gray', linewidth=0.5, linestyle=':')
    axes[1].set_title('Gross & Net Exposure', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Exposure')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title('N Positions', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Count')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    axes[3].set_title('Turnover (jours rebal)', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Turnover')
    axes[3].set_xlabel('Date')
    axes[3].legend(fontsize=9)
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_metrics_comparison(metrics_dict: dict) -> pd.DataFrame:
    """Tableau comparatif des métriques."""
    KEYS = [
        ('total_return',       'Total Return',   'pct'),
        ('annualized_return',  'CAGR',           'pct'),
        ('alpha_vs_rf',        'Alpha vs Rf',    'pct'),
        ('volatility_annual',  'Vol annualisee', 'pct'),
        ('sharpe_ratio',       'Sharpe',         'f3'),
        ('sortino_ratio',      'Sortino',        'f3'),
        ('calmar_ratio',       'Calmar',         'f3'),
        ('max_drawdown',       'Max Drawdown',   'pct'),
        ('var_95_hist',        'VaR 95%',        'pct'),
        ('cvar_95',            'CVaR 95%',       'pct'),
        ('beta',               'Beta',           'f4'),
        ('tracking_error',     'Tracking Error', 'pct'),
        ('information_ratio',  'Info Ratio',     'f3'),
        ('win_rate',           'Win Rate',       'pct'),
        ('profit_factor',      'Profit Factor',  'f2'),
    ]

    rows = []
    for key, label, fmt in KEYS:
        row = {'Metrique': label}
        for sname, m in metrics_dict.items():
            v = m.get(key, None)
            if v is None:
                row[sname] = '--'
            elif fmt == 'pct':
                row[sname] = f'{v*100:.2f}%'
            elif fmt == 'f2':
                row[sname] = f'{v:.2f}'
            elif fmt == 'f3':
                row[sname] = f'{v:.3f}'
            elif fmt == 'f4':
                row[sname] = f'{v:.4f}'
            else:
                row[sname] = f'{v}'
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df
