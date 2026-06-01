"""
visualisation.py — Reproduction des Figures 1‑5 du papier.

Toutes les fonctions retournent ``matplotlib.Figure`` et peuvent être
sauvegardées via ``fig.savefig(...)``.
"""
from __future__ import annotations

from typing import Mapping

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────── Figure 1 — graphes
def plot_portfolio_graphs(
    edges_baseline: pd.DataFrame,
    edges_matching: pd.DataFrame,
    title: str = "Figure 1 — Portfolio graphs",
    dark: bool = True,
):
    """Affiche le graphe Baseline (gauche) vs Matching (droite)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    if dark:
        fig.patch.set_facecolor("black")
    fig.suptitle(title, color="white" if dark else "black", fontsize=13)

    for ax, edges, label in [
        (axes[0], edges_baseline, f"Baseline\n({len(edges_baseline)} pairs)"),
        (axes[1], edges_matching, f"Matching\n({len(edges_matching)} pairs)"),
    ]:
        if dark:
            ax.set_facecolor("black")
        G = nx.Graph()
        for _, r in edges.iterrows():
            G.add_edge(r["asset_a"], r["asset_b"], weight=float(r["weight"]))
        if len(G):
            pos = nx.spring_layout(G, seed=42, k=0.4)
            nx.draw_networkx(
                G, pos=pos, ax=ax, with_labels=False, node_size=18,
                node_color="red", edge_color="#8844dd", width=0.7, alpha=0.85,
            )
        ax.set_title(label, color="white" if dark else "black", fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────── Figure 2 — cumulatif
def plot_cumulative_returns(
    series: Mapping[str, pd.Series],
    benchmark: pd.Series | None = None,
    title: str = "Figure 2 — Cumulative returns",
):
    fig, ax = plt.subplots(figsize=(11, 5))
    palette = {
        "Matching Q": ("#1f77b4", "-",  2.0),
        "Matching Z": ("#ff7f0e", "-",  2.0),
        "Baseline Q": ("#1f77b4", "--", 1.4),
        "Baseline Z": ("#ff7f0e", "--", 1.4),
    }
    for name, r in series.items():
        c = (1.0 + r.fillna(0.0)).cumprod() - 1.0
        color, ls, lw = palette.get(name, ("grey", "-", 1.2))
        ax.plot(c.index, c.values * 100, label=name, color=color, ls=ls, lw=lw)
    if benchmark is not None:
        cb = (1.0 + benchmark.fillna(0.0)).cumprod() - 1.0
        ax.plot(cb.index, cb.values * 100, label="S&P 500",
                color="mediumpurple", lw=1.1, alpha=0.8)
    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.set_title(title)
    ax.set_ylabel("Cumulative return (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────── Figure 3 — turnover
def plot_turnover_boxplot(
    matching: pd.Series, baseline: pd.Series,
    title: str = "Figure 3 — Monthly turnover",
):
    df = pd.DataFrame({"Matching": matching, "Baseline": baseline})
    fig, ax = plt.subplots(figsize=(6.5, 5))
    df.plot.box(ax=ax, patch_artist=True,
                color={"boxes": "steelblue", "medians": "crimson",
                       "whiskers": "steelblue", "caps": "steelblue"})
    for p in ax.patches:
        p.set_alpha(0.4)
    ax.set_title(title)
    ax.set_ylabel("Turnover (%)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────── Figure 4 — rétention
def plot_retention(
    matching: pd.Series, baseline: pd.Series,
    title: str = "Figure 4 — Jaccard retention",
):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(matching.index, matching.values, label="Matching",
            color="steelblue", lw=1.5)
    ax.plot(baseline.index, baseline.values, label="Baseline",
            color="darkorange", lw=1.5)
    ax.set_title(title)
    ax.set_ylabel("Retention (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────── Figure 5 — concentration
def plot_concentration(
    series: pd.Series, title: str = "Figure 5 — Stock concentration (Baseline)",
):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(series.index, series.values, color="steelblue", lw=1.3)
    ax.fill_between(series.index, series.values, alpha=0.15, color="steelblue")
    ax.set_title(title)
    ax.set_ylabel("Max single‑stock share (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig
