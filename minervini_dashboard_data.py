"""
minervini_dashboard_data.py
===========================
Read-only assembly of this week's Minervini setups for the dashboard's
/minervini tab (research/minervini/LIVE_PLAN.md). Same isolation pattern as
the other *_dashboard_data.py modules: its OWN read-only sqlite3 connection,
never research/minervini/scan.py's connect(). A setup is shown for five
sessions after its signal, then it leaves the tab.
"""
from __future__ import annotations

import sqlite3
from datetime import date
from typing import Dict, List, Optional

from config import MINERVINI_DB_PATH
from time_utils import previous_trading_day

SHOW_SESSIONS = 5

# state -> (words, dashboard.css badge)
STATES = {
    "waiting": ("Waiting for entry", "badge-neutral"),
    "bought": ("Entry hit", "badge-mild_bullish"),
    "target": ("Take profit hit", "badge-bullish"),
    "stopped": ("Loss point hit", "badge-bearish"),
    "expired": ("Not reached in a week", "badge-no_data"),
    "ran": ("Opened too high", "badge-no_data"),
}


def first_session_shown(session: str) -> str:
    day = date.fromisoformat(session)
    for _ in range(SHOW_SESSIONS):
        day = previous_trading_day(day)
    return day.isoformat()


def describe(r: Dict) -> Dict:
    words, badge = STATES.get(r["state"], (r["state"], "badge-neutral"))
    bought = r["state"] == "bought"
    entry = r["entry"] if r["entry"] is not None else r["pivot"]
    done = {"target": r["target"], "stopped": r["stop"]}.get(r["state"], r["last"] if bought else None)
    return {**r, "words": words, "badge": badge, "buy": entry, "limit": r["pivot"] * 1.05, "risk_pct": (r["stop"] / entry - 1) * 100,
            "distance_pct": (r["pivot"] / r["last"] - 1) * 100 if r["state"] == "waiting" and r["last"] else None,
            "result_pct": (done / entry - 1) * 100 if done else None}


def build_minervini_state() -> Dict:
    """{"rows": this week's setups, newest first, "session": last scanned session or None}"""
    if not MINERVINI_DB_PATH.exists():
        return {"rows": [], "session": None}
    conn = sqlite3.connect(f"file:{MINERVINI_DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        session: Optional[str] = conn.execute("SELECT MAX(session) FROM runs").fetchone()[0]
        if session is None:
            return {"rows": [], "session": None}
        rows: List[sqlite3.Row] = conn.execute("SELECT * FROM candidates WHERE signal >= ? ORDER BY signal DESC, rank DESC",
                                               (first_session_shown(session),)).fetchall()
    finally:
        conn.close()
    return {"rows": [describe(dict(r)) for r in rows], "session": session}
