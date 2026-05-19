from pathlib import Path
import pandas as pd

raw = Path(r"c:\Users\SebastianRochaext\OneDrive - Prompt Soluciones Integradas, S. de R.L. de C.V\Desktop\pairs_trading_novel\src\data\raw")
prices = pd.read_parquet(raw / "prices.parquet")
pe = pd.read_parquet(raw / "pe_ratios.parquet")
rf = pd.read_parquet(raw / "risk_free.parquet")
univ = pd.read_parquet(raw / "universe.parquet")

prices.index = pd.to_datetime(prices.index)
pe.index = pd.to_datetime(pe.index)
rf.index = pd.to_datetime(rf.index)
univ["date"] = pd.to_datetime(univ["date"])

print(f"prices_shape={prices.shape}")
print(f"prices_range={prices.index.min().date()}->{prices.index.max().date()}")
print(f"prices_missing_ratio={prices.isna().mean().mean():.4f}")

print(f"pe_shape={pe.shape}")
print(f"pe_range={pe.index.min().date()}->{pe.index.max().date()}")
print(f"pe_missing_ratio={pe.isna().mean().mean():.4f}")

print(f"rf_shape={rf.shape}")
print(f"rf_range={rf.index.min().date()}->{rf.index.max().date()}")
print(f"rf_col={rf.columns.tolist()}")

print(f"universe_shape={univ.shape}")
print(f"universe_range={univ['date'].min().date()}->{univ['date'].max().date()}")
print(f"universe_unique_dates={univ['date'].nunique()}")
print(f"universe_unique_tickers={univ['ticker'].nunique()}")

counts = univ.groupby('date')['ticker'].nunique()
print(f"universe_tickers_per_date=min:{int(counts.min())},median:{int(counts.median())},max:{int(counts.max())}")

price_t = set(prices.columns)
pe_t = set(pe.columns)
univ_t = set(univ['ticker'].unique())

print(f"tickers_in_prices={len(price_t)}")
print(f"tickers_in_pe={len(pe_t)}")
print(f"tickers_in_universe={len(univ_t)}")
print(f"prices_not_in_universe={len(price_t - univ_t)}")
print(f"pe_not_in_universe={len(pe_t - univ_t)}")
print(f"universe_not_in_prices={len(univ_t - price_t)}")
print(f"universe_not_in_pe={len(univ_t - pe_t)}")
