"""Offline tests for press_release_tracker.digest.build_digest (pure string
formatting, no network/DB)."""

from press_release_tracker.digest import SUBJECT_PREFIX, build_digest


def _row(**overrides):
    defaults = dict(
        guid="g1", feed_url="https://example.com/feed", title="Some Release",
        link="https://example.com/a.html", pubdate="Tue, 15 Sep 2026 08:36:00 GMT",
        ticker=None, company=None, category=None, materiality=None, summary=None,
    )
    defaults.update(overrides)
    return defaults


def test_build_digest_empty_rows():
    subject, body = build_digest([])
    assert subject.startswith(SUBJECT_PREFIX)
    assert "0 item" in subject
    assert body == "No new items."


def test_build_digest_subject_always_contains_the_shared_prefix():
    subject, _ = build_digest([_row()], kind="high")
    assert subject.startswith(SUBJECT_PREFIX)

    subject, _ = build_digest([_row()], kind="batch")
    assert subject.startswith(SUBJECT_PREFIX)


def test_build_digest_high_vs_batch_subject_wording():
    subject, _ = build_digest([_row()], kind="high")
    assert "HIGH" in subject

    subject, _ = build_digest([_row()], kind="batch")
    assert "hourly digest" in subject


def test_build_digest_singular_vs_plural_subject():
    subject, _ = build_digest([_row()])
    assert "1 item" in subject
    assert "1 items" not in subject

    subject, _ = build_digest([_row(guid="g1"), _row(guid="g2")])
    assert "2 items" in subject


def test_build_digest_uses_ticker_then_company_then_placeholder():
    _, body = build_digest([_row(ticker="OMI.V")])
    assert "[OMI.V]" in body

    _, body = build_digest([_row(ticker=None, company="Orosur Mining Inc.")])
    assert "[Orosur Mining Inc.]" in body

    _, body = build_digest([_row(ticker=None, company=None)])
    assert "[(ticker unknown)]" in body


def test_build_digest_includes_llm_fields_when_present():
    _, body = build_digest([_row(
        ticker="OMI.V", category="exploration_drilling", materiality="high",
        summary="Drilling results announced.",
    )])
    assert "category: exploration_drilling  materiality: high" in body
    assert "Drilling results announced." in body


def test_build_digest_omits_category_line_when_unparsed():
    _, body = build_digest([_row(category=None)])
    assert "category:" not in body
