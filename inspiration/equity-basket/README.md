# Equity Basket — Long-Short S&P 500

Backtest de deux stratégies long-short market-neutral (PCA Stat-Arb & Value EW Beta-Neutral) sur le S&P 500, 2001-2025.
Projet M2 IEF — **Sacha Guerin** & **Hugo Rocha**.

## Installation

Python >= 3.12 requis. Pour `blpapi`, le Bloomberg C SDK doit être installé et un terminal Bloomberg ouvert.

```bash
pip install -r requirements.txt
```

## Données

Les fichiers parquet sont attendus dans `data/raw/` :

- `prices.parquet` — prix ajustés quotidiens
- `pe_ratios.parquet` — P/E trailing
- `universe.parquet` — composition historique du S&P 500 (point-in-time)
- `risk_free.parquet` — T-Bill 3M

Si absents, le notebook propose de les télécharger depuis Bloomberg au lancement.

## Utilisation

Ouvrir `main.ipynb` et exécuter les cellules dans l'ordre. Tout est guidé par des prompts :

1. **Données** — chargement automatique depuis les parquets (ou téléchargement Bloomberg)
2. **Paramètres** — les résultats du grid search sont déjà sauvegardés (`outputs/gs_*.json`), répondre `N` pour les charger directement. Relancer le grid search prend ~10h.
3. **Backtests** — PCA Stat-Arb puis Value EW Beta-Neutral
4. **Comparaison** — overlay des equity curves et risk dashboards
5. **Exports** — XLSX importables dans Bloomberg PORT via BBU, CSV du benchmark, instructions de rebalancement
