"""
conviction_dashboard_data.py
=============================
Assembles the read model for the dashboard's Conviction tab from the
conviction_watchlist package -- a standalone personal-account tool (see
conviction_watchlist/__init__.py), NOT one of this repo's paper-trading
sleeves. Mirrors momentum_dashboard_data.py / macro_dashboard_data.py's own
pattern: a thin read/glue layer between a sleeve's own modules and the Flask
route in dashboard_app.py.
"""
from conviction_watchlist.entry_screener import load_last_result
from conviction_watchlist.quality_filter import load_cache, qualified_tickers
from conviction_watchlist.settings import load_settings
from conviction_watchlist.trailing_stop_monitor import compute_status


def build_conviction_view() -> dict:
    cache = load_cache()
    candidates_result = load_last_result()
    return {
        "settings": load_settings(),
        "quality_count": len(qualified_tickers(cache)),
        "quality_total_cached": len(cache),
        "quality_errors": sum(1 for r in cache.values() if "error" in r),
        "candidates_generated_at": candidates_result.get("generated_at"),
        "candidates": candidates_result.get("candidates", []),
        "holdings": compute_status(),
    }
