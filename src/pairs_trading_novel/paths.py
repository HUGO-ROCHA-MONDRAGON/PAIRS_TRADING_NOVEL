from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    root = Path(start or Path.cwd()).resolve()
    if root.is_file():
        root = root.parent
    while root != root.parent:
        if (root / "src").exists() and (root / "notebooks").exists():
            return root
        root = root.parent
    raise RuntimeError("Cannot locate project root containing src/ and notebooks/.")


def raw_data_dir(root: str | Path | None = None) -> Path:
    return find_project_root(root) / "src" / "data" / "raw"


def processed_data_dir(root: str | Path | None = None, phase: str = "phase1_full_replication") -> Path:
    out = find_project_root(root) / "src" / "data" / "processed" / phase
    out.mkdir(parents=True, exist_ok=True)
    return out


def required_raw_files(full: bool = True) -> list[str]:
    base = ["prices.parquet", "pe_ratios.parquet", "risk_free.parquet", "universe.parquet"]
    extra = ["benchmark.parquet", "log_prices.parquet", "daily_returns.parquet", "rebalance_dates.parquet"]
    return base + extra if full else base


def missing_raw_files(root: str | Path | None = None, full: bool = True) -> list[str]:
    raw = raw_data_dir(root)
    return [name for name in required_raw_files(full=full) if not (raw / name).exists()]
