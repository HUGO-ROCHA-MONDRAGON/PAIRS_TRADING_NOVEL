# Pairs Trading Replication (Graphical Matching) - Current Status and Roadmap

## 1) Objective

This repository is focused on replicating the paper:

**Pairs Trading Using a Novel Graphical Matching Approach**

The current project state is intentionally pragmatic:
- we proceed with the data we have today,
- we preserve methodological rigor,
- we document all approximations explicitly,
- we keep the pipeline ready to upgrade as additional data arrives.

---

## 2) What We Currently Have in the Repository

### Core folders

- `src/bloomberg/`
  - `blp.py`: low-level Bloomberg wrappers (`bdh`, `bdp`, `bds`) with batching, overrides, parsing, and adjustment controls.
  - `_session.py`: singleton Bloomberg session manager.
  - `provider.py`: high-level fetch/load/cache pipeline around the wrappers.
  - `__init__.py`: package exports and usage documentation.

- `src/data/raw/`
  - `prices.parquet`
  - `pe_ratios.parquet`
  - `risk_free.parquet`
  - `universe.parquet`

- `notebooks/`
  - `01_data_ingestion_graph_matching.ipynb`
  - `02_replication_partial_with_4_parquets.ipynb`

- `Pairs Trading Using a Novel Graphical Matching Approach.pdf`
  - Paper reference used to guide implementation.

---

## 3) What Has Already Been Done

## 3.1 Bloomberg access layer redesigned

A new Bloomberg access package was implemented under `src/bloomberg/`, inspired by xbbg-style ergonomics and robustness:

- Unified request functions:
  - `bdh` for historical data
  - `bdp` for reference data
  - `bds` for bulk data

- Key technical improvements:
  - direct Bloomberg overrides through keyword arguments
  - request batching for larger ticker sets
  - shared event collection logic for partial/final Bloomberg responses
  - normalized output formats for downstream use
  - safe handling of missing field exceptions

- Provider-level features:
  - point-in-time index membership retrieval
  - adjusted prices retrieval
  - PE ratio retrieval
  - risk-free retrieval
  - benchmark retrieval
  - static metadata retrieval
  - parquet cache save/load and fetch-or-load orchestration

## 3.2 Data migration and baseline data inventory

Historical raw data was reused and centralized in `src/data/raw/`.

Available datasets are currently limited to 4 parquet files (see section 4).

## 3.3 Data ingestion notebook created

`notebooks/01_data_ingestion_graph_matching.ipynb` provides:
- project-root safe imports,
- configuration and data window setup,
- provider load/fetch entrypoint,
- quality checks,
- base feature generation (log prices, returns),
- export of canonical intermediate datasets.

## 3.4 Partial replication notebook created

`notebooks/02_replication_partial_with_4_parquets.ipynb` implements a full Phase-1 research pipeline with constrained data:

- monthly rebalance loop from historical membership snapshots,
- value-filtered candidate universe using PE where available,
- pair scoring with Engle-Granger style residual stationarity proxy,
- disjoint pair selection via greedy max-weight matching approximation,
- monthly spread trading simulation with z-score entry/exit/stop,
- diagnostics, retention statistics, and result export.

## 3.5 Robustness fixes already integrated

Important practical fixes were added while validating the pipeline:

- safer date filtering in formation windows (no brittle label slicing assumptions),
- safe handling when PE columns are missing for part of the active universe,
- fallback logic when value-ranked candidates are insufficient,
- stronger resilience for sparse/misaligned cross-sectional data.

---

## 4) Data We Have Right Now (Ground Truth)

The current replication can only use these 4 files:

1. `prices.parquet`
   - shape: `(6539, 1248)`
   - date range: `2000-01-03` to `2025-12-31`
   - average missing ratio: `0.3475`

2. `pe_ratios.parquet`
   - shape: `(6539, 1214)`
   - date range: `2000-01-03` to `2025-12-31`
   - average missing ratio: `0.4045`

3. `risk_free.parquet`
   - shape: `(6783, 1)`
   - column: `risk_free`
   - date range: `2000-01-03` to `2025-12-31`

4. `universe.parquet`
   - shape: `(156564, 2)`
   - columns: `date`, `ticker`
   - date range: `2000-01-31` to `2025-12-31`
   - unique dates: `312`
   - unique tickers: `1249`
   - tickers per date: min `499`, median `500`, max `506`

Ticker coverage consistency snapshot:
- tickers in prices: `1248`
- tickers in PE: `1214`
- tickers in universe: `1249`
- prices not in universe: `15`
- PE not in universe: `13`
- universe not in prices: `16`
- universe not in PE: `48`

This mismatch is expected in historical point-in-time equity data and is handled by the notebook logic.

---

## 5) What Is Replicable Today vs What Is Not Yet

## 5.1 Replicable now (partial but operational)

With current data and code, we can already run:

- rolling formation windows,
- pair candidate generation from point-in-time universe,
- residual-based stationarity ranking (proxy t-stat),
- disjoint pair portfolio construction,
- z-score mean-reversion signal simulation,
- portfolio-level diagnostics and persistence.

## 5.2 Not yet fully paper-faithful

The following are still approximations or missing pieces:

- Exact ADF test implementation with proper lag treatment and p-values.
- Exact graph maximum-weight matching (current notebook uses greedy disjoint matching).
- Full benchmark handling in all workflows (partial notebook currently uses a proxy where benchmark parquet is unavailable).
- Realistic transaction cost and market-friction model.
- Full execution assumptions and detailed portfolio constraints as in production backtests.

---

## 6) How to Run the Current Pipeline

## Step A: Ingestion and baseline checks

Run `notebooks/01_data_ingestion_graph_matching.ipynb` from top to bottom.

Purpose:
- validate data paths and schema,
- build canonical derived datasets,
- ensure base feature readiness.

## Step B: Partial replication experiment

Run `notebooks/02_replication_partial_with_4_parquets.ipynb` from top to bottom.

Purpose:
- execute the constrained replication engine,
- generate strategy diagnostics,
- export outputs for analysis and iteration.

Expected output directory after execution:
- `src/data/processed/phase1_partial_replication/`

Expected artifacts:
- `diagnostics.csv`
- `candidate_edges.parquet` (if non-empty)
- `selected_pairs.parquet` (if non-empty)
- `strategy_returns.parquet` (if non-empty)
- `pair_retention.csv` (if non-empty)

---

## 7) Assumptions and Current Research Conventions

To keep progress rigorous with constrained data, we currently assume:

- universe membership is monthly and point-in-time from `universe.parquet`,
- candidate selection may rely on PE but falls back to coverage-ranked assets,
- pair quality is ranked using residual stationarity proxy t-stat,
- portfolio uses disjoint pairs and equal aggregation,
- trading logic is z-score spread mean reversion with entry/exit/stop thresholds,
- risk-free conversion uses heuristic normalization depending on value scale.

These assumptions are documented so they can be replaced incrementally.

---

## 8) Recommended Next Steps (Priority Roadmap)

## Priority 1 - Methodological parity upgrades

1. Replace greedy disjoint pairing with exact maximum-weight matching.
2. Replace ADF proxy with exact ADF test (including lag policy and p-values).
3. Ensure edge-weight definition matches paper equations exactly.

## Priority 2 - Data completeness upgrades

1. Add benchmark total return series and integrate consistently.
2. Add static metadata and sector labels in the active research pipeline.
3. Add missing Bloomberg fields needed for deeper paper diagnostics.

## Priority 3 - Backtest realism upgrades

1. Add transaction costs and turnover-linked slippage.
2. Add leverage and capital allocation rules aligned with paper setup.
3. Add execution-delay assumptions and rebalance timing controls.

## Priority 4 - Validation and robustness framework

1. Build a parameter sensitivity grid (entry/exit/stop, pair count, ADF threshold).
2. Add walk-forward and subperiod stability checks.
3. Add ablation analysis (value filter on/off, universe restrictions, matching variants).

## Priority 5 - Reproducibility and engineering hardening

1. Add deterministic config files for experiment runs.
2. Add logging and run manifests for each experiment output.
3. Add tests for data integrity, pair construction, and signal logic.

---

## 9) Definition of Done for Full Replication

We can consider the replication complete when all of the following hold:

1. Data inputs and transformations match the paper specification.
2. Pair formation and matching method match paper methodology exactly.
3. Signal and execution rules match paper assumptions.
4. Main reported metrics are reproducible in the target sample.
5. Robustness checks confirm stability and no hidden implementation bias.

---

## 10) Current Bottom Line

The project is no longer at setup stage: it is in active research execution stage.

- Infrastructure exists.
- Data pipeline exists.
- Partial replication engine exists.
- Known gaps are explicit and actionable.

The next milestone is to upgrade from constrained baseline replication to method-faithful replication while preserving reproducibility.
