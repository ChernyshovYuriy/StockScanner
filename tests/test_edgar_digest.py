"""Offline tests for the EDGAR digest builder and email_log dedup (no network)."""

from edgar.digest import build_digest, FOOTER
from edgar import store


# ── digest builder ───────────────────────────────────────────────────────────

def test_quiet_day_returns_none():
    assert build_digest("2026-06-05", [], []) is None


def test_digest_has_sections_counts_and_footer():
    insiders = [{"ticker": "MU", "owner": "DOE JANE", "shares": 3000.0,
                 "price": 9.5, "date": "2026-06-05"}]
    activist = [{"ticker": "XYZ", "cik": 1, "form": "SC 13D",
                 "date": "2026-06-05", "url": "https://sec.gov/x"}]
    subject, body = build_digest("2026-06-05", insiders, activist)

    assert "1 insider buy," in subject
    assert "1 activist" in subject
    assert "INSIDER BUYS" in body and "ACTIVIST STAKES" in body
    assert "MU" in body and "XYZ" in body
    assert "13D" in body
    assert FOOTER in body


def test_subject_pluralizes_insider_count():
    subject, _ = build_digest(
        "2026-06-05", [], [{"ticker": "A", "form": "SC 13G", "date": "d", "url": "u"}]
    )
    assert "0 insider buys," in subject


def test_insider_only_digest_omits_activist_section():
    insiders = [{"ticker": "KEY", "owner": "X", "shares": 100, "price": 10.0, "date": "d"}]
    subject, body = build_digest("2026-06-05", insiders, [])
    assert "INSIDER BUYS" in body
    assert "ACTIVIST STAKES" not in body


# ── email_log dedup ──────────────────────────────────────────────────────────

def test_email_log_quiet_then_sent(tmp_path):
    conn = store.connect(tmp_path / "edgar.db")

    # Nothing recorded yet.
    assert store.already_sent(conn, "2026-06-05") is False

    # A quiet day (sent=0) must NOT count as sent — a later run can still email.
    store.record_email(conn, "2026-06-05", 0, sent=0)
    assert store.already_sent(conn, "2026-06-05") is False

    # Once a real digest goes out, a same-day re-run is blocked.
    store.record_email(conn, "2026-06-05", 3, sent=True)
    assert store.already_sent(conn, "2026-06-05") is True

    # Independent dates are tracked separately.
    assert store.already_sent(conn, "2026-06-08") is False
