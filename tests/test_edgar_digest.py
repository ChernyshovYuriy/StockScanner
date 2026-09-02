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


def test_insider_buy_cluster_marker_appears_when_flagged():
    insiders = [{"ticker": "MU", "owner": "DOE JANE", "shares": 1000.0,
                 "price": 5.0, "date": "2026-06-05", "cluster": True}]
    _, body = build_digest("2026-06-05", insiders, [])
    assert ">> CLUSTER" in body


def test_insider_buy_without_cluster_flag_has_no_marker():
    insiders = [{"ticker": "MU", "owner": "DOE JANE", "shares": 1000.0,
                 "price": 5.0, "date": "2026-06-05"}]
    _, body = build_digest("2026-06-05", insiders, [])
    assert ">> CLUSTER" not in body


def test_insider_buy_single_lot_has_no_rollup_line():
    insiders = [{"ticker": "MU", "owner": "DOE JANE", "shares": 1000.0,
                 "price": 5.0, "date": "2026-06-05"}]
    _, body = build_digest("2026-06-05", insiders, [])
    assert "lots" not in body


def test_insider_buy_multiple_lots_same_day_get_a_rollup_line():
    """Two lots by the same insider, same name, same day: each lot still
    gets its own line, plus one rollup totalling shares and value."""
    insiders = [
        {"ticker": "MU", "owner": "DOE JANE", "shares": 1000.0, "price": 5.0,
         "date": "2026-06-05"},
        {"ticker": "MU", "owner": "DOE JANE", "shares": 500.0, "price": 6.0,
         "date": "2026-06-05"},
    ]
    _, body = build_digest("2026-06-05", insiders, [])
    assert body.count("DOE JANE") == 3   # two lot lines + the rollup line
    assert "2 lots, 1,500 sh total, $8,000" in body


def test_new_insider_form3_appears_in_digest_and_subject():
    new_insiders = [{"ticker": "MU", "cik": 723125, "date": "2026-06-05",
                      "url": "https://sec.gov/x"}]
    subject, body = build_digest("2026-06-05", [], [], new_insiders)
    assert "1 new insider" in subject
    assert "NEW INSIDERS" in body and "MU" in body


def test_new_insider_alone_is_not_a_quiet_day():
    """A lone Form 3, with no buys or activist hits, must still trigger a
    digest rather than being silently treated as nothing happened."""
    new_insiders = [{"ticker": "MU", "cik": 723125, "date": "2026-06-05",
                      "url": "https://sec.gov/x"}]
    assert build_digest("2026-06-05", [], [], new_insiders) is not None


def test_empty_new_insiders_list_is_still_a_quiet_day():
    assert build_digest("2026-06-05", [], [], []) is None


def test_no_new_insiders_omits_that_section_and_subject_clause():
    subject, body = build_digest(
        "2026-06-05", [{"ticker": "MU", "owner": "X", "shares": 1, "price": 1.0, "date": "d"}],
        [], [],
    )
    assert "new insider" not in subject
    assert "NEW INSIDERS" not in body


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
