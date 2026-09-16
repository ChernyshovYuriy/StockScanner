"""Offline tests for press_release_tracker.llm_parser (no real OpenAI call --
openai.OpenAI is always monkeypatched to a fake client)."""

import json

import openai

from press_release_tracker import llm_parser


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content=None, exc=None):
        self._content = content
        self._exc = exc

    def create(self, **kwargs):
        if self._exc:
            raise self._exc
        return _FakeCompletion(self._content)


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeOpenAI:
    def __init__(self, content=None, exc=None):
        self.chat = _FakeChat(_FakeCompletions(content, exc))

    def __call__(self, api_key=None):
        return self


def test_parse_release_returns_none_without_api_key(monkeypatch):
    monkeypatch.setattr(llm_parser, "OPENAI_API_KEY", "")
    assert llm_parser.parse_release("title", "desc", []) is None


def test_parse_release_returns_structured_dict_on_success(monkeypatch):
    monkeypatch.setattr(llm_parser, "OPENAI_API_KEY", "sk-test")
    payload = json.dumps({
        "ticker": "OMI.V", "company": "Orosur Mining Inc.",
        "category": "exploration_drilling", "materiality": "high",
        "summary": "Drilling results announced.",
    })
    monkeypatch.setattr(openai, "OpenAI", lambda api_key: _FakeOpenAI(content=payload))

    result = llm_parser.parse_release("Orosur announces drilling results", "desc text", ["Mining"])

    assert result == {
        "ticker": "OMI.V", "company": "Orosur Mining Inc.",
        "category": "exploration_drilling", "materiality": "high",
        "summary": "Drilling results announced.",
    }


def test_parse_release_defaults_missing_keys(monkeypatch):
    monkeypatch.setattr(llm_parser, "OPENAI_API_KEY", "sk-test")
    payload = json.dumps({"ticker": None, "company": None})
    monkeypatch.setattr(openai, "OpenAI", lambda api_key: _FakeOpenAI(content=payload))

    result = llm_parser.parse_release("title", "desc", [])

    assert result["category"] == "other"
    assert result["materiality"] == "low"
    assert result["summary"] == ""


def test_parse_release_returns_none_on_api_exception(monkeypatch):
    monkeypatch.setattr(llm_parser, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(openai, "OpenAI", lambda api_key: _FakeOpenAI(exc=RuntimeError("boom")))

    assert llm_parser.parse_release("title", "desc", []) is None


def test_parse_release_returns_none_on_malformed_json(monkeypatch):
    monkeypatch.setattr(llm_parser, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(openai, "OpenAI", lambda api_key: _FakeOpenAI(content="not json"))

    assert llm_parser.parse_release("title", "desc", []) is None
