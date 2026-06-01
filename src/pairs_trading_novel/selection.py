"""
selection.py — Sélecteurs de portefeuille (Baseline vs Matching).

Baseline : top‑N paires par poids décroissant (peut réutiliser le même actif).
Matching : matching maximum‑weight exact sur le graphe non‑orienté
           (algorithme d'Edmonds, via networkx.max_weight_matching).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import networkx as nx
import pandas as pd

from .core import select_baseline, select_matching


class PairSelector(ABC):
    """Interface commune. ``max_pairs`` borne la taille du portefeuille."""

    def __init__(self, max_pairs: int = 25) -> None:
        self.max_pairs = max_pairs

    @abstractmethod
    def select(self, edges: pd.DataFrame) -> pd.DataFrame: ...

    def __call__(self, edges: pd.DataFrame) -> pd.DataFrame:
        return self.select(edges)


class BaselineSelector(PairSelector):
    """Top‑N par t‑stat le plus négatif (≡ poids le plus élevé)."""

    def select(self, edges: pd.DataFrame) -> pd.DataFrame:
        return select_baseline(edges, max_pairs=self.max_pairs)


class MatchingSelector(PairSelector):
    """Maximum‑weight matching exact ; aucun actif n'apparaît dans 2 paires."""

    def select(self, edges: pd.DataFrame) -> pd.DataFrame:
        return select_matching(edges, max_pairs=self.max_pairs)

    @staticmethod
    def build_graph(edges: pd.DataFrame) -> nx.Graph:
        graph = nx.Graph()
        for _, row in edges.iterrows():
            graph.add_edge(
                row["asset_a"], row["asset_b"], weight=float(row["weight"])
            )
        return graph
