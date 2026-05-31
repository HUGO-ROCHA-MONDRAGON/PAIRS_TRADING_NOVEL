# Pairs Trading Novel — Graphical Matching Replication

This repository contains a cleaned, GitHub-ready research pipeline for replicating **Pairs Trading Using a Novel Graphical Matching Approach** on a point-in-time S&P 500 universe.

The project focuses on a reproducible workflow:

1. Bloomberg data ingestion and parquet cache creation.
2. Formation-window pair scoring using Engle-Granger style residual stationarity.
3. Graph-based maximum-weight matching versus a greedy baseline.
4. Monthly mean-reversion backtest with z-score trading rules.
5. Diagnostics, return series and performance tables.

## Repository layout

```text
.
├── notebooks/
│   ├── 01_data_ingestion_graph_matching.ipynb
│   ├── 02_replication_partial_with_4_parquets.ipynb
│   └── 03_full_paper_replication.ipynb
├── scripts/
│   ├── execute_notebook.py
│   ├── inspect_data.py
│   └── run_full_backtest.py
├── src/
│   ├── bloomberg/
│   ├── pairs_trading_novel/
│   └── data/
│       ├── raw/
│       └── processed/
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

## Data policy

This public-ready archive intentionally excludes raw Bloomberg data, processed parquet outputs, local virtual environments and debug files.

To reproduce the full backtest, place the following files in `src/data/raw/`:

```text
prices.parquet
pe_ratios.parquet
risk_free.parquet
universe.parquet
benchmark.parquet
log_prices.parquet
daily_returns.parquet
rebalance_dates.parquet
```

The full private reproducibility archive can keep these files locally, but they should not be committed to a public GitHub repository unless you have the right to distribute the data.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

For Bloomberg ingestion, install and configure Bloomberg `blpapi` separately, keep Bloomberg Terminal open, then run the ingestion notebook.

## Quick checks

```bash
python -m compileall src scripts
python scripts/inspect_data.py
```

`inspect_data.py` requires the parquet data files to be present.

## Run the pipeline

### Step 1 — Data ingestion

Run:

```bash
jupyter notebook notebooks/01_data_ingestion_graph_matching.ipynb
```

This validates or downloads the raw datasets and creates canonical derived files such as log-prices, daily returns and rebalance dates.

### Step 2 — Baseline constrained replication

Run:

```bash
jupyter notebook notebooks/02_replication_partial_with_4_parquets.ipynb
```

This version only needs the four minimal parquet files: prices, P/E ratios, risk-free rate and point-in-time universe.

### Step 3 — Full replication

Either run the notebook:

```bash
jupyter notebook notebooks/03_full_paper_replication.ipynb
```

or run the CLI script:

```bash
python scripts/run_full_backtest.py
```

Main outputs are written to:

```text
src/data/processed/phase1_full_replication/
```

Expected outputs include:

```text
diagnostics.csv
candidate_edges.parquet
selected_pairs_matching.parquet
selected_pairs_baseline.parquet
strategy_returns.parquet
table_performance.csv
```

## Methodology snapshot

Current implementation:

- point-in-time S&P 500 monthly universe,
- 504-trading-day formation window,
- OLS residual spread estimation in both orientations,
- fast ADF(1) t-statistic proxy on residuals,
- exact maximum-weight matching through NetworkX,
- greedy top-weight baseline,
- z-score entry/exit/stop trading simulation,
- one-way transaction-cost parameter.

Important approximation:

- `CORR_THRESHOLD = 0.50` is used as a computational pre-filter. Setting it to `0.0` makes the candidate search closer to the exhaustive paper setting, but much slower.

## GitHub push checklist

```bash
git init
git add .
git commit -m "Initial cleaned replication pipeline"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```

Before pushing publicly, confirm that no proprietary data or copyrighted PDFs are tracked:

```bash
git status
git ls-files src/data
```

## Guides en français

Des résumés en français, concis et pratiques pour les notebooks principaux ont été ajoutés :
- [01 — Ingestion et préparation des données (FR)](notebooks/01_data_ingestion_graph_matching_fr.md)
- [02 — Réplication partielle avec 4 Parquets (FR)](notebooks/02_replication_partial_with_4_parquets_fr.md)
- [03 — Réplication complète du papier (FR)](notebooks/03_full_paper_replication_fr.md)

Ces documents donnent des instructions rapides, des conseils pratiques et les commandes d'exécution recommandées.
