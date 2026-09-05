"""
Read/write helpers for config.HOLDINGS_FILE -- the user's manually-maintained
list of real positions this sleeve tracks. Nothing here places, infers, or
simulates a trade; every add/remove is a direct action from the dashboard's
Conviction tab (or a hand-edit of the JSON file itself).
"""
import json

from conviction_watchlist.config import HOLDINGS_FILE


def load_holdings() -> list:
    if not HOLDINGS_FILE.exists():
        return []
    return json.loads(HOLDINGS_FILE.read_text())


def save_holdings(holdings: list) -> None:
    HOLDINGS_FILE.parent.mkdir(exist_ok=True)
    HOLDINGS_FILE.write_text(json.dumps(holdings, indent=2))


def add_holding(ticker: str, entry_date: str, entry_price: float, qty: float, account: str) -> None:
    holdings = load_holdings()
    holdings.append({
        "ticker": ticker.strip().upper(),
        "entry_date": entry_date,
        "entry_price": entry_price,
        "qty": qty,
        "account": account.strip(),
    })
    save_holdings(holdings)


def remove_holding(index: int) -> None:
    holdings = load_holdings()
    if 0 <= index < len(holdings):
        holdings.pop(index)
        save_holdings(holdings)
