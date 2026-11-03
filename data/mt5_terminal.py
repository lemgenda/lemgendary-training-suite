"""MetaTrader 5 terminal access for the forex data pipeline.

Connection lifecycle, historical bar download and synthetic mock bars for offline runs.
"""

from __future__ import annotations

import ctypes
import importlib
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("lemtrain.data.mt5_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Constants (instrument universe and timeframes live in data.forex_spec)
# MetaTrader 5 Timeframe Attribute Mapping
MT5_TIMEFRAMES = {
    1: "TIMEFRAME_M1",
    5: "TIMEFRAME_M5",
    15: "TIMEFRAME_M15",
    60: "TIMEFRAME_H1",
    240: "TIMEFRAME_H4",
    1440: "TIMEFRAME_D1",
}


def _terminal_window_present() -> bool:
    """Return True when a "MetaTrader 5" window exists (always False off Windows)."""
    if sys.platform != "win32":
        return False
    try:
        hwnd = ctypes.windll.user32.FindWindowW(None, "MetaTrader 5")
        if not hwnd:
            print(
                " [MT5] MetaTrader 5 terminal window not detected. Operating in offline/mock mode."
            )
            return False
    except (AttributeError, OSError) as exc:
        # AttributeError: ctypes.windll unavailable; OSError: user32 cannot be loaded.
        logger.warning("MT5 terminal window probe failed; operating offline: %s", exc)
        return False
    return True


def _attach_running_session(mt5: Any) -> bool:
    """Attach to the running terminal session; True when it reports an account."""
    try:
        if mt5.initialize():
            info = mt5.account_info()
            if info:
                print(
                    f" [MT5] Connected -> Account: {info.login} | Server: {info.server} | "
                    f"Balance: {info.balance} {info.currency}"
                )
                return True
    except (RuntimeError, OSError) as exc:
        logger.warning(
            "Direct attach to running MT5 terminal failed; trying explicit login: %s", exc
        )
    return False


def connect_mt5(
    login: int | None = None,
    password: str | None = None,
    server: str | None = None,
    api_key: str | None = None,
) -> bool:
    """
    Initialize MetaTrader 5 connection.

    Credentials can be passed explicitly, or fall back to environment variables:
        MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_API_KEY

    Returns:
        True if connection succeeded, False otherwise.
    """
    if login is None and "MT5_LOGIN" in os.environ:
        try:
            login = int(os.environ["MT5_LOGIN"])
        except ValueError as exc:
            logger.warning("Ignoring non-integer MT5_LOGIN environment value: %s", exc)
    if password is None:
        password = os.environ.get("MT5_PASSWORD")
    if server is None:
        server = os.environ.get("MT5_SERVER")
    if api_key is None:
        api_key = os.environ.get("MT5_API_KEY")

    try:
        mt5: Any = importlib.import_module("MetaTrader5")
    except ImportError:
        print(" [MT5] MetaTrader5 package not installed. Run: pip install MetaTrader5")
        print(" [MT5] NOTE: MT5 package is Windows-only and requires MT5 terminal installed.")
        return False

    # Fast non-blocking check: only attempt IPC attach if the terminal runs or MT5_ENABLE=1
    if os.environ.get("MT5_ENABLE", "0") != "1" and not _terminal_window_present():
        return False

    # First try attaching directly to the running MT5 terminal session
    if _attach_running_session(mt5):
        return True

    # Fallback to explicit login parameters if active session not found
    init_kwargs = {}
    if login and password and server:
        init_kwargs.update({"login": login, "password": password, "server": server})

    try:
        if not mt5.initialize(**init_kwargs):
            err = mt5.last_error()
            print(f" [MT5] Initialize failed: {err}")
            return False
    except (RuntimeError, OSError, TypeError, ValueError) as e:
        # TypeError/ValueError: rejected login/password/server arguments.
        logger.error("MT5 initialize raised: %s", e)
        print(f" [MT5] Initialize exception: {e}")
        return False

    info = mt5.account_info()
    if info:
        print(
            f" [MT5] Connected -> Account: {info.login} | Server: {info.server} | "
            f"Balance: {info.balance} {info.currency}"
        )
    else:
        print(" [MT5] Connected to MT5 terminal session.")
    return True


def disconnect_mt5():
    """Cleanly shut down the MT5 connection."""
    try:
        mt5: Any = importlib.import_module("MetaTrader5")
    except ImportError as exc:
        logger.debug("MetaTrader5 not installed; nothing to disconnect: %s", exc)
        return
    try:
        mt5.shutdown()
        print(" [MT5] Disconnected.")
    except (RuntimeError, OSError) as exc:
        logger.warning("MT5 shutdown failed: %s", exc)


def generate_mock_bars(
    pair: str, timeframe_min: int, n_bars: int | None = None, start_date: str = "2019-01-01"
) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV bars spanning 2019-01-01 to present for all 16 assets."""
    np.random.seed(abs(hash(pair + str(timeframe_min))) % (2**32 - 1))

    price_map = {
        "EURUSD": 1.0850,
        "GBPUSD": 1.2700,
        "USDJPY": 155.00,
        "XAUUSD": 2400.00,
        "USDCAD": 1.3650,
        "USDCHF": 0.8950,
        "AUDUSD": 0.6650,
        "NZDUSD": 0.6100,
        "EURJPY": 168.00,
        "GBPJPY": 196.50,
        "EURGBP": 0.8550,
        "XAGUSD": 29.50,
        "USOIL": 78.50,
        "US500": 5500.0,
        "USTEC": 19500.0,
        "GER40": 18500.0,
    }
    base_price = price_map.get(pair, 1.0000)

    if any(x in pair for x in ["JPY", "XAG"]):
        pip_size = 0.01
    elif any(x in pair for x in ["XAU", "USOIL"]):
        pip_size = 0.1
    elif any(x in pair for x in ["US500", "USTEC", "GER40"]):
        pip_size = 1.0
    else:
        pip_size = 0.0001

    dt_end = datetime.now(timezone.utc)
    freq = f"{timeframe_min}min" if timeframe_min < 1440 else "1D"

    if n_bars is not None and n_bars > 0 and start_date is None:
        times = pd.date_range(end=dt_end, periods=n_bars, freq=freq)
    else:
        start_dt = (
            pd.to_datetime(start_date, utc=True)
            if start_date
            else pd.to_datetime("2019-01-01", utc=True)
        )
        times = pd.date_range(start=start_dt, end=dt_end, freq=freq)
        if n_bars is not None and len(times) < n_bars:
            times = pd.date_range(end=dt_end, periods=n_bars, freq=freq)

    total_bars = len(times)
    returns = np.random.normal(loc=0.0, scale=0.001, size=total_bars)
    price_curve = base_price * np.exp(np.cumsum(returns))

    spread = pip_size * np.random.uniform(1.0, 3.0, size=total_bars)
    highs = price_curve + np.abs(np.random.normal(0, pip_size * 5, size=total_bars)) + spread
    lows = price_curve - np.abs(np.random.normal(0, pip_size * 5, size=total_bars)) - spread
    opens = price_curve + np.random.uniform(-spread, spread, size=total_bars)
    closes = price_curve

    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))
    volumes = np.random.randint(100, 5000, size=total_bars)

    print(
        f" [MT5Pipeline] Synthesized {total_bars} bars for {pair} {timeframe_min}min "
        f"(Spans {times[0].strftime('%Y-%m-%d')} -> {times[-1].strftime('%Y-%m-%d')})"
    )
    return pd.DataFrame(
        {
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


def download_bars(
    pair: str, timeframe_min: int, n_bars: int = 50000, start_date: str = "2019-01-01"
) -> pd.DataFrame:
    """
    Download historical OHLCV bars from MT5 for a given pair and timeframe.

    Args:
        pair:          Symbol (e.g. "EURUSD")
        timeframe_min: Timeframe in minutes (1, 5, 15, 60, 240, 1440)
        n_bars:        Number of bars to download
        start_date:    Start date string (e.g. "2019-01-01")

    Returns:
        DataFrame with columns: time, open, high, low, close, volume
    """
    try:
        mt5: Any = importlib.import_module("MetaTrader5")
        tf_attr = MT5_TIMEFRAMES.get(timeframe_min)
        if tf_attr is None:
            raise ValueError(f"Unsupported timeframe: {timeframe_min}min")

        tf = getattr(mt5, tf_attr)
        rates = None
        if start_date:
            try:
                dt_from = pd.to_datetime(start_date, utc=True).to_pydatetime()
                dt_to = datetime.now(timezone.utc)
                rates = mt5.copy_rates_range(pair, tf, dt_from, dt_to)
            except (ValueError, TypeError, RuntimeError) as exc:
                # ValueError: unparseable start_date;
                # TypeError/RuntimeError: MT5 rejected the range query.
                logger.warning(
                    "copy_rates_range failed for %s %smin; falling back to copy_rates_from_pos: %s",
                    pair,
                    timeframe_min,
                    exc,
                )
                rates = None

        if rates is None or len(rates) == 0:
            rates = mt5.copy_rates_from_pos(pair, tf, 0, n_bars)

        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            raise RuntimeError(f"No data returned for {pair} {timeframe_min}min: {error}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["volume"] = df["tick_volume"]
        df = df[["time", "open", "high", "low", "close", "volume"]]
        span_start = pd.Series(df["time"]).iloc[0].strftime("%Y-%m-%d")
        span_end = pd.Series(df["time"]).iloc[-1].strftime("%Y-%m-%d")
        print(
            f" [MT5] Downloaded {len(df)} bars for {pair} {timeframe_min}min "
            f"(Spans {span_start} -> {span_end})"
        )
        assert isinstance(df, pd.DataFrame)
        return df

    except ImportError as exc:
        raise RuntimeError(
            "[MT5] MetaTrader5 not installed. Run: pip install MetaTrader5\n"
            "      Ensure MT5 terminal is running with a demo account."
        ) from exc
