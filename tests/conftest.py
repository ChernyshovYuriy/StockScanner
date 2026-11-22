import pytest


@pytest.fixture(autouse=True)
def _no_dashboard_prefetch(monkeypatch):
    """The dashboard views' batched Yahoo prefetch (dashboard_market_data.py)
    would hit the network in every view test; those tests stub the per-ticker
    loaders instead. tests/test_dashboard_market_data.py tests it directly."""
    import dashboard_market_data
    monkeypatch.setattr(dashboard_market_data, "prefetch", lambda daily, intraday=(): None)
    dashboard_market_data._daily.clear()
    dashboard_market_data._intraday.clear()
    yield
    dashboard_market_data._daily.clear()
    dashboard_market_data._intraday.clear()
