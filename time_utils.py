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


# ── TSX regular session + holiday calendar ───────────────────────────────────
# TSX regular trading session, in minutes past midnight Eastern Time.
_MARKET_OPEN_MINUTES = 9 * 60 + 30   # 09:30 ET
_MARKET_CLOSE_MINUTES = 16 * 60      # 16:00 ET

# TSX statutory holidays when the exchange is fully closed (observed dates;
# weekend holidays roll forward to the next business day).  Maintain yearly —
# an out-of-date list only weakens the holiday layer; the weekday + session-hours
# checks in is_market_open() still block off-hours runs regardless.
TSX_HOLIDAYS: frozenset = frozenset({
    # 2026
    "2026-01-01",  # New Year's Day
    "2026-02-16",  # Family Day
    "2026-04-03",  # Good Friday
    "2026-05-18",  # Victoria Day
    "2026-07-01",  # Canada Day
    "2026-08-03",  # Civic Holiday
    "2026-09-07",  # Labour Day
    "2026-10-12",  # Thanksgiving
    "2026-12-25",  # Christmas Day
    "2026-12-28",  # Boxing Day (observed; Dec 26 is a Saturday)
    # 2027
    "2027-01-01",  # New Year's Day
    "2027-02-15",  # Family Day
    "2027-03-26",  # Good Friday
    "2027-05-24",  # Victoria Day
    "2027-07-01",  # Canada Day
    "2027-08-02",  # Civic Holiday
    "2027-09-06",  # Labour Day
    "2027-10-11",  # Thanksgiving
    "2027-12-27",  # Christmas Day (observed; Dec 25 is a Saturday)
    "2027-12-28",  # Boxing Day (observed; Dec 26 is a Sunday)
})


def is_market_open(tz: ZoneInfo = TSX_TZ) -> bool:
    """
    Return True only when the TSX is in its regular session *right now*:
    a weekday, within 09:30-16:00 ET, and not a TSX holiday.

    Used as a safety guard so the scheduled services never transact on stale
    data when triggered outside trading hours (e.g. a systemd Persistent
    catch-up, a manual run, or a holiday).  Routes through market_now(), so it
    is deterministic under set_backtest_clock().
    """
    now = market_now(tz)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    if now.strftime(ISO_DATE_EXTENDED) in TSX_HOLIDAYS:
        return False
    minutes = now.hour * 60 + now.minute
    return _MARKET_OPEN_MINUTES <= minutes < _MARKET_CLOSE_MINUTES


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
