from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Centralized formats
TSX_TZ = ZoneInfo("America/Toronto")
ISO_DATE_BASIC = "%Y%m%d"
ISO_DATE_BASIC_MINUTES = ISO_DATE_BASIC + "T%H%M"
ISO_DATE_EXTENDED = "%Y-%m-%d"


def market_now(tz: ZoneInfo = TSX_TZ) -> datetime:
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
