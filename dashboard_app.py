"""
dashboard_app.py
=================
Read/write web dashboard for the trading system: shows live open positions
(Monitor), trade history + transaction ledger (History), and lets a human
close a position at market price via manual_sell.sell_position().

LAN-only, no authentication — deliberate choice for a home-network Jetson
deployment; anyone on the LAN can view and sell.

Local dev:
    python dashboard_app.py
    curl localhost:8080/healthz

Deployed on the Jetson via system/stockscanner-dashboard.service, which runs
this same entrypoint under Waitress (pure-Python WSGI server, no C-extension
build step on ARM).
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any, Callable

import duckdb
from flask import Flask, jsonify, redirect, render_template, request, url_for

from config import DASHBOARD_HOST, DASHBOARD_PORT
from conviction_dashboard_data import build_conviction_view
from conviction_watchlist import quality_filter as conviction_quality_filter
from conviction_watchlist.entry_screener import refresh_and_save as conviction_refresh_candidates
from conviction_watchlist.holdings_store import add_holding as conviction_add_holding
from conviction_watchlist.holdings_store import remove_holding as conviction_remove_holding
from conviction_watchlist.settings import save_settings as conviction_save_settings
from dashboard_positions import build_live_positions
from db import get_all_trades, get_cash, get_transactions
from demand_dashboard_data import build_demand_signals_by_ticker
from demand_signals.summary import summarize_all
from macro_dashboard_data import build_macro_positions, get_current_regime, get_macro_cash, get_macro_transactions
from manual_sell import sell_position
from momentum_dashboard_data import build_momentum_positions, get_momentum_cash, get_momentum_transactions

_ERROR_STATUS = {
    "locked": 409,
    "no_position": 404,
    "no_price": 503,
    "already_closed": 409,
}


def _compute_initial_capital(cash: float, transactions) -> float:
    """Derive starting capital from the current cash balance and the
    all-time BUY/SELL ledger: cash_now = initial - buys + sells, so
    initial = cash_now + buys - sells. db.py has no separate "initial
    capital" field — set_cash() is only ever called once, at first-time
    setup (see CLAUDE.md), so this holds as long as cash was never manually
    adjusted again after go-live."""
    if transactions.empty:
        return cash
    buys = transactions.loc[transactions["side"] == "BUY", "amount"].sum()
    sells = transactions.loc[transactions["side"] == "SELL", "amount"].sum()
    return cash + buys - sells


def _read_with_retry(fn: Callable[[], Any], retries: int = 2, backoff: float = 0.3) -> Any:
    """Best-effort retry for a DB read that lands on DuckDB's writer lock
    while a scheduled service (main/buy/monitor) is mid-write. db.py itself
    is left unchanged; this wrapper only exists here because the dashboard
    reads far more often (continuous polling) than the old world (4 short
    runs a day)."""
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except (duckdb.Error, OSError) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff)
    raise last_exc  # noqa: B904 — re-raising the last captured DB error as-is


def create_app() -> Flask:
    """Build the Flask app. Does not call init_db() itself — the caller
    (the __main__ entrypoint below, or a test fixture) is responsible for
    that, since db.init_db()'s bare (no-arg) form always resets DB_PATH back
    to the production default, which would clobber a test's tmp_path DB if
    called again in here."""
    app = Flask(__name__)

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True})

    @app.get("/")
    def monitor():
        try:
            rows = _read_with_retry(build_live_positions)
            cash = _read_with_retry(get_cash)
            transactions = _read_with_retry(get_transactions)
            error = None
        except (duckdb.Error, OSError):
            rows, cash, transactions = [], None, None
            error = "Database temporarily unavailable — retrying on next refresh."

        total_pnl = None
        market_value = None
        initial_capital = None
        total_equity = None
        total_return = None
        total_return_pct = None

        if rows:
            total_pnl = sum(
                row["pnl_$"] for row in rows if isinstance(row.get("pnl_$"), (int, float))
            )
            market_value = sum(
                row["last_close"] * row["shares"] for row in rows
                if isinstance(row.get("last_close"), (int, float)) and isinstance(row.get("shares"), (int, float))
            )
        if cash is not None:
            total_equity = cash + (market_value or 0.0)
        if cash is not None and transactions is not None:
            initial_capital = _compute_initial_capital(cash, transactions)
        if total_equity is not None and initial_capital:
            total_return = total_equity - initial_capital
            total_return_pct = (total_equity / initial_capital - 1.0) * 100.0

        return render_template(
            "monitor.html",
            rows=rows,
            cash=cash,
            total_pnl=total_pnl,
            market_value=market_value,
            initial_capital=initial_capital,
            total_equity=total_equity,
            total_return=total_return,
            total_return_pct=total_return_pct,
            error=error,
        )

    @app.get("/momentum")
    def momentum():
        """Read-only view of the momentum sleeve (separate DB/capital — see
        config.py MOMENTUM_* and momentum_dashboard_data.py). No sell action
        here: db.py's global DB_PATH means a manual-sell route sharing this
        process with the core sleeve's would race between the two DBs."""
        try:
            rows = _read_with_retry(build_momentum_positions)
            cash = _read_with_retry(get_momentum_cash)
            transactions = _read_with_retry(get_momentum_transactions)
            error = None
        except (duckdb.Error, OSError):
            rows, cash, transactions = [], None, None
            error = "Database temporarily unavailable — retrying on next refresh."

        total_pnl = None
        market_value = None
        initial_capital = None
        total_equity = None
        total_return = None
        total_return_pct = None

        if rows:
            total_pnl = sum(
                row["pnl_$"] for row in rows if isinstance(row.get("pnl_$"), (int, float))
            )
            market_value = sum(
                row["last_close"] * row["shares"] for row in rows
                if isinstance(row.get("last_close"), (int, float)) and isinstance(row.get("shares"), (int, float))
            )
        if cash is not None:
            total_equity = cash + (market_value or 0.0)
        if cash is not None and transactions is not None:
            initial_capital = _compute_initial_capital(cash, transactions)
        if total_equity is not None and initial_capital:
            total_return = total_equity - initial_capital
            total_return_pct = (total_equity / initial_capital - 1.0) * 100.0

        return render_template(
            "momentum_monitor.html",
            rows=rows,
            cash=cash,
            total_pnl=total_pnl,
            market_value=market_value,
            initial_capital=initial_capital,
            total_equity=total_equity,
            total_return=total_return,
            total_return_pct=total_return_pct,
            error=error,
        )

    @app.get("/macro")
    def macro():
        """Read-only view of the macro conviction sleeve (separate DB/capital
        — see config.py MACRO_* and macro_dashboard_data.py). No sell action
        here, same isolation reasoning as /momentum."""
        try:
            rows = _read_with_retry(build_macro_positions)
            cash = _read_with_retry(get_macro_cash)
            transactions = _read_with_retry(get_macro_transactions)
            error = None
        except (duckdb.Error, OSError):
            rows, cash, transactions = [], None, None
            error = "Database temporarily unavailable — retrying on next refresh."

        # get_current_regime() makes a live HTTP call (through macro_regime.py's
        # CachedClient, cache-keyed by day) — _read_with_retry is typed for
        # duckdb.Error/OSError, not requests exceptions, so this gets its own
        # try/except rather than reusing that wrapper.
        try:
            regime = get_current_regime()
        except Exception:
            regime = {"label": "unknown", "composite": None, "votes": {}, "detail": {"error": "unavailable"}}

        total_pnl = None
        market_value = None
        initial_capital = None
        total_equity = None
        total_return = None
        total_return_pct = None

        if rows:
            total_pnl = sum(
                row["pnl_$"] for row in rows if isinstance(row.get("pnl_$"), (int, float))
            )
            market_value = sum(
                row["last_close"] * row["shares"] for row in rows
                if isinstance(row.get("last_close"), (int, float)) and isinstance(row.get("shares"), (int, float))
            )
        if cash is not None:
            total_equity = cash + (market_value or 0.0)
        if cash is not None and transactions is not None:
            initial_capital = _compute_initial_capital(cash, transactions)
        if total_equity is not None and initial_capital:
            total_return = total_equity - initial_capital
            total_return_pct = (total_equity / initial_capital - 1.0) * 100.0

        return render_template(
            "macro_monitor.html",
            rows=rows,
            cash=cash,
            total_pnl=total_pnl,
            market_value=market_value,
            initial_capital=initial_capital,
            total_equity=total_equity,
            total_return=total_return,
            total_return_pct=total_return_pct,
            regime=regime,
            error=error,
        )

    @app.get("/demand")
    def demand():
        """Read-only view of demand_signals.db (EDGAR insider buys + FINRA
        dark-pool ratio + options-flow proxy, normalized — see
        demand_signals/__init__.py). No action here, same as /momentum: this
        is a display layer over what demand_signals_service.py has already
        populated, never a trigger for a fetch or a trade."""
        try:
            rows = _read_with_retry(build_demand_signals_by_ticker)
            error = None
        except (sqlite3.Error, OSError):
            rows = {}
            error = "Database temporarily unavailable — retrying on next refresh."

        summaries = summarize_all(rows)
        return render_template("demand_signals.html", rows=rows, summaries=summaries, error=error)

    @app.get("/conviction")
    def conviction():
        """Read/write view of the conviction_watchlist package -- a
        standalone personal tool for the user's real RBC account (Margin +
        TFSA), NOT one of the paper-trading sleeves above. No DB, no
        db.py/config.py (root) involvement at all -- see
        conviction_watchlist/__init__.py."""
        try:
            view = build_conviction_view()
            error = request.args.get("error")
        except Exception as e:
            view = {"settings": {}, "quality_count": 0, "quality_total_cached": 0, "quality_errors": 0,
                    "candidates_generated_at": None, "candidates": [], "holdings": []}
            error = f"Failed to load conviction watchlist state: {e}"
        return render_template("conviction.html", error=error, **view)

    @app.post("/conviction/settings")
    def conviction_settings():
        try:
            updates = {
                "dip_pct_off_high": float(request.form["dip_pct_off_high"]) / 100.0,
                "trailing_stop_pct": float(request.form["trailing_stop_pct"]) / 100.0,
                "min_market_cap_cad": float(request.form["min_market_cap_cad"]),
                "min_price": float(request.form["min_price"]),
            }
            conviction_save_settings(updates)
        except (KeyError, ValueError) as e:
            return redirect(url_for("conviction", error=f"Bad settings input: {e}"))
        return redirect(url_for("conviction"))

    @app.post("/conviction/holdings/add")
    def conviction_holdings_add():
        try:
            conviction_add_holding(
                ticker=request.form["ticker"],
                entry_date=request.form["entry_date"],
                entry_price=float(request.form["entry_price"]),
                qty=float(request.form["qty"]),
                account=request.form.get("account", ""),
            )
        except (KeyError, ValueError) as e:
            return redirect(url_for("conviction", error=f"Bad holding input: {e}"))
        return redirect(url_for("conviction"))

    @app.post("/conviction/holdings/<int:index>/remove")
    def conviction_holdings_remove(index: int):
        conviction_remove_holding(index)
        return redirect(url_for("conviction"))

    @app.post("/conviction/refresh-quality")
    def conviction_refresh_quality():
        try:
            conviction_quality_filter.rebuild()
        except Exception as e:
            return redirect(url_for("conviction", error=f"Quality list refresh failed: {e}"))
        return redirect(url_for("conviction"))

    @app.post("/conviction/refresh-candidates")
    def conviction_refresh_candidates_route():
        try:
            conviction_refresh_candidates()
        except Exception as e:
            return redirect(url_for("conviction", error=f"Entry screener refresh failed: {e}"))
        return redirect(url_for("conviction"))

    @app.get("/history")
    def history():
        try:
            trades = _read_with_retry(get_all_trades).to_dict("records")
            transactions = _read_with_retry(get_transactions).to_dict("records")
            error = None
        except (duckdb.Error, OSError):
            trades, transactions = [], []
            error = "Database temporarily unavailable — try again shortly."
        return render_template(
            "history.html",
            trades=trades,
            transactions=transactions,
            error=error,
        )

    @app.post("/api/positions/<ticker>/sell")
    def sell(ticker: str):
        body = request.get_json(silent=True) or {}
        price = body.get("price")
        if price is not None:
            try:
                price = float(price)
            except (TypeError, ValueError):
                return jsonify({"ok": False, "ticker": ticker, "error": "bad_price",
                                 "message": "Price must be a number."}), 400
            if price <= 0:
                return jsonify({"ok": False, "ticker": ticker, "error": "bad_price",
                                 "message": "Price must be positive."}), 400

        result = sell_position(ticker, price=price)
        status = 200 if result["ok"] else _ERROR_STATUS.get(result["error"], 500)
        return jsonify(result), status

    return app


if __name__ == "__main__":
    from waitress import serve

    from db import init_db

    init_db()
    serve(create_app(), host=DASHBOARD_HOST, port=DASHBOARD_PORT)
