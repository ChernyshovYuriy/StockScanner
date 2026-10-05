"""Offline tests for press_release_tracker.article / analyst (no network --
the article fetch and openai.OpenAI are always monkeypatched)."""

import json

import openai

from press_release_tracker import analyst, article, store

_HTML = """<html><body><nav>Menu stuff</nav>
<div class="main-body-container" itemprop="articleBody">
<p>Acme Corp. (TSXV: ACME) reports Q3 results.</p>
<script>var x = 1;</script>
<table><tr><td>Revenue</td><td>$1.2M</td><td></td><td>$0.8M</td></tr>
<tr><th>Net loss</th><td>(0.3M)</td></tr></table>
<p>Cash of $4.0M.<br>No debt.</p>
</div><footer>Footer text</footer></body></html>"""


def test_extract_article_text_keeps_body_and_tables_only():
    text = article.extract_article_text(_HTML)
    assert "Acme Corp. (TSXV: ACME) reports Q3 results." in text
    assert "Revenue | $1.2M | $0.8M" in text
    assert "Net loss | (0.3M)" in text
    assert "Cash of $4.0M.\nNo debt." in text
    assert "Menu stuff" not in text and "Footer text" not in text and "var x" not in text


def test_extract_article_text_none_without_body_marker():
    assert article.extract_article_text("<html><p>no body here</p></html>") is None


def test_should_analyze_rules():
    fin = {"ticker": "ACME.V", "category": "financing", "materiality": "low"}
    assert analyst.should_analyze(fin, "https://www.globenewswire.com/news-release/2026/10/01/3/0/en/x.html")
    assert not analyst.should_analyze(fin, "https://www.globenewswire.com/news-release/2026/10/01/3/0/fr/x.html")
    assert not analyst.should_analyze({**fin, "ticker": None}, "https://x/0/en/y")
    assert not analyst.should_analyze(None, "https://x")
    other = {"ticker": "ACME.V", "category": "personnel", "materiality": "medium"}
    assert not analyst.should_analyze(other, "https://x")
    assert analyst.should_analyze({**other, "materiality": "high"}, "https://x")


class _FakeOpenAI:
    def __init__(self, content=None, exc=None):
        self.content, self.exc, self.kwargs = content, exc, None
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.kwargs = kwargs
                if outer.exc:
                    raise outer.exc

                class _Msg:
                    content = outer.content

                class _Choice:
                    message = _Msg()

                class _Resp:
                    choices = [_Choice()]
                return _Resp()

        class _Chat:
            completions = _Completions()
        self.chat = _Chat()

    def __call__(self, api_key=None):
        return self


def test_analyze_release_none_without_key(monkeypatch):
    monkeypatch.setattr(analyst, "OPENAI_API_KEY", "")
    assert analyst.analyze_release("ACME.V", None, "t", "https://x", body="text", context=None) is None


def test_analyze_release_none_when_article_unreadable(monkeypatch):
    monkeypatch.setattr(analyst, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(analyst.article, "fetch_article_text", lambda link: None)
    assert analyst.analyze_release("ACME.V", None, "t", "https://x") is None


def test_analyze_release_sends_body_and_context_and_returns_result(monkeypatch):
    monkeypatch.setattr(analyst, "OPENAI_API_KEY", "sk-test")
    fake = _FakeOpenAI(content=json.dumps({"verdict": "bullish", "trend": "improving"}))
    monkeypatch.setattr(openai, "OpenAI", fake)
    ctx = {"yahoo_ticker": "ACME.V", "price": 0.5, "market_cap": 20e6, "shares_outstanding": 40e6,
           "currency": "CAD"}
    out = analyst.analyze_release("ACME.V", "Acme", "Acme Q3", "https://x", body="Revenue up", context=ctx)
    assert out["verdict"] == "bullish"
    assert out["body_chars"] == len("Revenue up")
    assert out["market_context"] == ctx
    user = fake.kwargs["messages"][1]["content"]
    assert "Revenue up" in user
    assert "market cap 20.0M" in user and "shares outstanding 40.0M" in user


def test_analyze_release_none_on_api_error_or_bad_json(monkeypatch):
    monkeypatch.setattr(analyst, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI(exc=RuntimeError("boom")))
    assert analyst.analyze_release("A", None, "t", "u", body="b", context=None) is None
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI(content="not json"))
    assert analyst.analyze_release("A", None, "t", "u", body="b", context=None) is None


def test_save_analysis_and_unemailed_carries_it(tmp_path):
    from press_release_tracker.feeds import FeedItem

    conn = store.connect(tmp_path / "pr.db")
    store.mark_seen(conn, FeedItem("g1", "f", "T", "l", "p", "d", []), first_seen_at="2026-10-03T09:00:00")
    store.mark_seen(conn, FeedItem("g2", "f", "T2", "l2", "p", "d", []), first_seen_at="2026-10-03T09:01:00")
    a = {"verdict": "neutral", "trend": "stable", "confidence": "low",
         "dilution": {"level": "none", "detail": ""}, "body_chars": 5, "release_type": "financing"}
    store.save_analysis(conn, "g1", "ACME.V", a, "gpt-5-mini", "2026-10-03T09:02:00")
    rows = store.unemailed(conn)
    assert rows[0]["analysis"] == a
    assert rows[1]["analysis"] is None
    assert conn.execute("SELECT dilution_level, release_type FROM release_analysis").fetchone() == ("none", "financing")


def test_format_analysis_skips_absent_dilution():
    lines = analyst.format_analysis({"verdict": "bullish", "trend": "improving", "confidence": "high",
                                     "verdict_reason": "r", "key_figures": ["Revenue +50%"],
                                     "dilution": {"level": "none", "detail": "no shares"}})
    text = "\n".join(lines)
    assert "ANALYST: BULLISH" in text and "Revenue +50%" in text
    assert "dilution" not in text
