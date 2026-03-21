from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

# Centralized formats
TSX_TZ = ZoneInfo("America/Toronto")
ISO_DATE_BASIC = "%Y%m%d"
ISO_DATE_BASIC_MINUTES = ISO_DATE_BASIC + "T%H%M"
ISO_DATE_EXTENDED = "%Y-%m-%d"

# ── Backtest clock ────────────────────────────────────────────────────────────
# In live/paper mode this is always None and market_now() returns the real wall
# clock — identical to the original behaviour.
# In backtest mode, call set_backtest_clock(sim_datetime) before each simulated
# day; all callers of market_now() / market_today() will see the injected time
# without any other changes required.
_backtest_now: Optional[datetime] = None


def set_backtest_clock(dt: Optional[datetime]) -> None:
    """
    Set the simulated "current time" for backtest mode.

    Pass a timezone-aware datetime to fix the clock at a historical moment.
    Pass None (default) to restore live wall-clock behaviour.

    Example:
        from time_utils import set_backtest_clock, TSX_TZ
        from datetime import datetime
        set_backtest_clock(datetime(2025, 3, 15, 16, 5, tzinfo=TSX_TZ))
        # ... run one simulated day ...
        set_backtest_clock(None)  # restore live mode
    """
    global _backtest_now
    if dt is not None and dt.tzinfo is None:
        raise ValueError(
            "set_backtest_clock() requires a timezone-aware datetime. "
            "Use e.g. datetime(..., tzinfo=TSX_TZ)."
        )
    _backtest_now = dt


def is_backtest_mode() -> bool:
    """Return True when the clock is pinned to a historical datetime."""
    return _backtest_now is not None


def market_now(tz: ZoneInfo = TSX_TZ) -> datetime:
    if _backtest_now is not None:
        return _backtest_now.astimezone(tz).replace(microsecond=0)
    return datetime.now(timezone.utc).astimezone(tz).replace(microsecond=0)


def market_today(tz: ZoneInfo = TSX_TZ) -> datetime:
    return market_now(tz).replace(hour=0, minute=0, second=0, microsecond=0)


def market_today_str(fmt: str = ISO_DATE_EXTENDED, tz: ZoneInfo = TSX_TZ) -> str:
    return market_today(tz).strftime(fmt)


def start_of_market_day(tz: ZoneInfo = TSX_TZ) -> datetime:
    d = market_today(tz)
    return datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=tz)


def date_to_iso_basic(date: datetime) -> str:
    return date.strftime(ISO_DATE_BASIC)


def date_to_iso_basic_minutes(date: datetime) -> str:
    return date.strftime(ISO_DATE_BASIC_MINUTES)


def date_to_iso_extended(date: datetime) -> str:
    return date.strftime(ISO_DATE_EXTENDED)


if __name__ == "__main__":
    print(f"Market now        '{market_now()}'")
    print(f"Markey today      '{market_today()}'")
    print(f"Markey today date '{market_today().date()}'")
    print(f"Markey today str  '{market_today_str()}'")
    print(f"Markey start day  '{start_of_market_day()}'")
    print(f"To ISO basic      '{date_to_iso_basic(market_now())}'")
    print(f"To ISO basic min  '{date_to_iso_basic_minutes(market_now())}'")
    print(f"To ISO extended   '{date_to_iso_extended(market_now())}'")
    print(f"Debug             '{datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)}'")
