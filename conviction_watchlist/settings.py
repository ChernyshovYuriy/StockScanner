"""
Runtime-adjustable settings for conviction_watchlist -- the knobs exposed as
"controls" on the dashboard's Conviction tab (dip %, trailing-stop %, quality
bar). Falls back to the fixed defaults in config.py whenever
data/conviction_settings.json doesn't exist or is missing a key, so the CLI
scripts keep working with sane values even if the dashboard has never been
used to change anything.
"""
import json

from conviction_watchlist import config

SETTINGS_FILE = config.REPO_ROOT / "data" / "conviction_settings.json"

DEFAULTS = {
    "dip_pct_off_high": config.DIP_PCT_OFF_HIGH,
    "trailing_stop_pct": config.TRAILING_STOP_PCT,
    "min_market_cap_cad": config.MIN_MARKET_CAP_CAD,
    "min_price": config.MIN_PRICE,
}


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            saved = json.loads(SETTINGS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            saved = {}
    else:
        saved = {}
    return {**DEFAULTS, **saved}


def save_settings(updates: dict) -> dict:
    merged = {**load_settings(), **updates}
    SETTINGS_FILE.parent.mkdir(exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(merged, indent=2))
    return merged
