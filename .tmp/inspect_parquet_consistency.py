from pathlib import Path
import pandas as pd

base = Path(r"c:\Users\SebastianRochaext\OneDrive - Prompt Soluciones Integradas, S. de R.L. de C.V\Desktop\pairs_trading_novel\src\data\raw")
prices = pd.read_parquet(base / "prices.parquet")
pe = pd.read_parquet(base / "pe_ratios.parquet")
rf = pd.read_parquet(base / "risk_free.parquet")
univ = pd.read_parquet(base / "universe.parquet")

univ['date'] = pd.to_datetime(univ['date'])

print(f"universe_date_min={univ['date'].min()} | universe_date_max={univ['date'].max()}")
print(f"universe_unique_dates={univ['date'].nunique()} | universe_unique_tickers={univ['ticker'].nunique()}")
counts = univ.groupby('date')['ticker'].nunique()
print(f"tickers_per_date_min={counts.min()} | median={counts.median():.0f} | max={counts.max()}")

price_t = set(prices.columns)
pe_t = set(pe.columns)
univ_t = set(univ['ticker'].unique())
print(f"tickers_in_prices={len(price_t)} | in_pe={len(pe_t)} | in_universe={len(univ_t)}")
print(f"prices_not_in_universe={len(price_t - univ_t)}")
print(f"pe_not_in_universe={len(pe_t - univ_t)}")
print(f"universe_not_in_prices={len(univ_t - price_t)}")
print(f"universe_not_in_pe={len(univ_t - pe_t)}")

prices_idx = pd.to_datetime(prices.index)
pe_idx = pd.to_datetime(pe.index)
rf_idx = pd.to_datetime(rf.index)
print(f"prices_rows={len(prices_idx)} | dates={prices_idx.min()} -> {prices_idx.max()}")
print(f"pe_rows={len(pe_idx)} | dates={pe_idx.min()} -> {pe_idx.max()}")
print(f"rf_rows={len(rf_idx)} | dates={rf_idx.min()} -> {rf_idx.max()}")
