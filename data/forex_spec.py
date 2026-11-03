"""Forex instrument universe, timeframe ladder and on-disk shard layout.

Single source of truth shared by the MT5 data pipeline, ``ForexDataset``,
``ForexPredictor`` and the MT5 signal exporter.
"""

from __future__ import annotations

import os

# Timeframe rungs in minutes; mirrors the Governor's res_ladder.
TIMEFRAME_RUNGS = [1, 5, 15, 60, 240, 1440]

# Canonical lookback windows per timeframe (covers ~1 trading week each).
TIMEFRAME_LOOKBACK = {
    1: 512,  # M1  → ~8.5 hours
    5: 288,  # M5  → ~1 day
    15: 192,  # M15 → ~2 days
    60: 168,  # H1  → ~1 week
    240: 90,  # H4  → ~2.5 weeks
    1440: 252,  # D1  → ~1 year
}

# Currency universe: Titan 4 starting core, extensible up to 16 professional assets.
TITAN_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
MAJOR_PAIRS = TITAN_PAIRS

EXTENDED_PAIRS = [
    # G7 Majors (4 Core + 4 G7)
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "XAUUSD",
    "USDCAD",
    "USDCHF",
    "AUDUSD",
    "NZDUSD",
    # High-Beta Crosses
    "EURJPY",
    "GBPJPY",
    "EURGBP",
    # Commodities & Energy
    "XAGUSD",
    "USOIL",
    # Global Equity Indices
    "US500",
    "NAS100",
    "DE40",
]
NUM_PAIRS = len(EXTENDED_PAIRS)

# Alternate CFD broker tickers -> canonical instrument name.
ALIAS_PAIRS = {
    "USTEC": "NAS100",
    "GER40": "DE40",
}

# Symbols requested from the MetaTrader 5 terminal: the broker ticker where one exists.
_BROKER_TICKERS = {canonical: alias for alias, canonical in ALIAS_PAIRS.items()}
MT5_SYMBOLS = [_BROKER_TICKERS.get(pair, pair) for pair in EXTENDED_PAIRS]

# Currency pair index (0-based); broker aliases share their canonical index.
PAIR_INDEX = {pair: i for i, pair in enumerate(EXTENDED_PAIRS)}
PAIR_INDEX.update({alias: PAIR_INDEX[canonical] for alias, canonical in ALIAS_PAIRS.items()})

# Canonical pip scaling factors (Normalized Pip Units / NPU): scales disparate market pip
# magnitudes into a standardized ~[10, 100] range. Aliases share the canonical factor.
PAIR_PIP_SCALE = {
    # FX majors, JPY pairs and high-beta crosses
    "EURUSD": 1.0,
    "GBPUSD": 1.0,
    "USDJPY": 1.0,
    "USDCAD": 1.0,
    "USDCHF": 1.0,
    "AUDUSD": 1.0,
    "NZDUSD": 1.0,
    "EURGBP": 1.0,
    "EURJPY": 1.0,
    "GBPJPY": 1.0,
    # Commodities and precious metals
    "XAGUSD": 5.0,
    "USOIL": 5.0,
    "XAUUSD": 10.0,
    # Global indices (move in thousands of raw index points)
    "US500": 20.0,
    "DE40": 20.0,
    "NAS100": 40.0,
}
PAIR_PIP_SCALE.update(
    {alias: PAIR_PIP_SCALE[canonical] for alias, canonical in ALIAS_PAIRS.items()}
)


def shard_file_paths(shard_dir: str, chunk_idx: int | None = None) -> tuple[str, str, str, str]:
    """Return the (X, y_dir, y_mag, timestamps) ``.npy`` paths of a shard or shard chunk."""
    suffix = f"_chunk{chunk_idx}" if chunk_idx is not None and chunk_idx >= 0 else ""
    return (
        os.path.join(shard_dir, f"X{suffix}.npy"),
        os.path.join(shard_dir, f"y_dir{suffix}.npy"),
        os.path.join(shard_dir, f"y_mag{suffix}.npy"),
        os.path.join(shard_dir, f"timestamps{suffix}.npy"),
    )
