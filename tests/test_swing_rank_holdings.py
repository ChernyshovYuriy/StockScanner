"""Swing rank -- point-in-time index membership. No network."""
from __future__ import annotations

import pandas as pd
import pytest

from research.swing_rank import holdings as hd

HEAD = ('﻿Fund Holdings as of,"{date}"\n \n'
        'Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Shares,Price,Location,Exchange,Currency,FX Rate,Market Currency\n')


def row(t, price="10.00", shares="1,000.00", cls="Equity", exch="Toronto Stock Exchange", sector="Energy", cur="CAD"):
    return f'"{t}","{t} INC","{sector}","{cls}","1","0.50","1","{shares}","{price}","Canada","{exch}","{cur}","1.00","CAD"\n'


def doc(date="Mar 31, 2020", n=200, extra=""):
    return HEAD.format(date=date) + "".join(row(f"T{k}") for k in range(n)) + extra + '\n"The content..."\n'


def test_parse_reads_date_members_and_numbers():
    as_of, m = hd.parse(doc(extra=row("RY", "1,287.17", "4,022,296.00", sector="Financials")))
    assert as_of == pd.Timestamp("2020-03-31") and len(m) == 201
    r = m.set_index("ticker").loc["RY"]
    assert r["price"] == 1287.17 and r["shares"] == 4022296 and r["sector"] == "Financials" and r["weight"] == 0.5


def test_parse_drops_cash_foreign_and_priceless_rows():
    extra = (row("CAD", cls="Cash") + row("FUT", cls="Futures") + row("US", cur="USD")
             + row("DEAD", price="0.00") + row("NOPX", price="-") + row("NOSH", shares="-"))
    _, m = hd.parse(doc(n=3, extra=extra))
    assert m["ticker"].tolist() == ["T0", "T1", "T2"]


def test_parse_keeps_interlisted_members_whatever_the_exchange_label():
    """2006-2016 files call RY's exchange New York; the price is still CAD."""
    _, m = hd.parse(doc(n=1, extra=row("RY", exch="New York Stock Exchange Inc.") + row("945", exch="Hong Kong Exchanges And Clearing Ltd")))
    assert m["ticker"].tolist() == ["T0", "RY", "945"] and m["exchange"].iloc[1].startswith("New York")


def test_parse_skips_the_footer_and_short_rows():
    assert len(hd.parse(doc(n=2, extra='"The content","x"\n'))[1]) == 2


def test_parse_duplicate_row_counted_once():
    assert len(hd.parse(doc(n=3, extra=row("T1")))[1]) == 3


def test_parse_keeps_two_companies_under_one_ticker():
    """Older files label both IAMGOLD and Industrial Alliance IAG."""
    other = row("T1").replace("T1 INC", "SOMEONE ELSE")
    assert len(hd.parse(doc(n=3, extra=other))[1]) == 4


@pytest.mark.parametrize("text", ["", "<html>blocked</html>", 'Fund Holdings as of,"-"\n', 'x,"y"'])
def test_parse_garbage_is_empty(text):
    as_of, m = hd.parse(text)
    assert as_of is None and m.empty


def test_month_ends_are_weekdays_and_complete_months_only():
    e = hd.month_ends("2020-01", pd.Timestamp("2020-06-29"))
    assert [d.strftime("%m%d") for d in e] == ["0131", "0228", "0331", "0430", "0529"]


class Site:
    def __init__(self, closed=(), dead_after=None, wrong_date=False, small=False):
        self.closed, self.dead_after, self.wrong, self.small, self.asked = set(closed), dead_after, wrong_date, small, []

    def __call__(self, day):
        self.asked.append(day)
        if day in self.closed or (self.dead_after and len(self.asked) > self.dead_after):
            return ""
        d = pd.Timestamp(day) if not self.wrong else pd.Timestamp("2026-11-20")
        return doc(d.strftime("%b %d, %Y"), 20 if self.small else 200)


def run(tmp_path, site, last="2020-04-15", first="2020-01"):
    naps = []
    out = hd.fetch(pd.Timestamp(last), first, tmp_path, site, naps.append, lambda m: None)
    assert len(naps) == len(site.asked) and set(naps) <= {hd.PAUSE_SECONDS}       # every request is paced
    return out


def test_fetch_saves_each_month_once(tmp_path):
    assert run(tmp_path, Site()) == {"fetched": 3, "missing": 0}
    assert sorted(p.name for p in tmp_path.glob("*")) == ["20200131.csv", "20200228.csv", "20200331.csv"]
    again = Site()
    assert run(tmp_path, again) == {"fetched": 0, "missing": 0} and again.asked == []


def test_fetch_steps_back_over_a_holiday(tmp_path):
    site = Site(closed=["20200331"])
    run(tmp_path, site)
    assert (tmp_path / "20200330.csv").exists() and "20200331" in site.asked


def test_single_dead_months_do_not_stop_the_run(tmp_path):
    dead = [d.strftime("%Y%m%d") for m in ("2020-01", "2020-03", "2020-05", "2020-07") for d in pd.bdate_range(m, periods=23) if d.strftime("%Y-%m") == m]
    out = run(tmp_path, Site(closed=dead), last="2020-08-31")
    assert out == {"fetched": 4, "missing": 4}


def test_fetch_rejects_a_file_dated_otherwise(tmp_path):
    with pytest.raises(hd.FetchFailed):
        run(tmp_path, Site(wrong_date=True))
    assert list(tmp_path.glob("*")) == []


def test_fetch_rejects_a_tiny_file(tmp_path):
    with pytest.raises(hd.FetchFailed):
        run(tmp_path, Site(small=True))


def test_fetch_stops_after_three_dead_months(tmp_path):
    site = Site(dead_after=2)
    with pytest.raises(hd.FetchFailed):
        run(tmp_path, site, last="2021-12-31")
    assert len(list(tmp_path.glob("*"))) == 2 and len(site.asked) == 2 + 3 * hd.MAX_BACK_DAYS


def test_published_gap_is_skipped_not_fatal(tmp_path):
    empty = 'Fund Holdings as of,"-"\nTicker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Shares,Price,Location,Exchange,Currency,FX Rate,Market Currency\n'

    class Gappy(Site):
        def __call__(self, day):
            self.asked.append(day)
            return empty if "20200201" <= day <= "20200631" else super().__call__(day)
    out = hd.fetch(pd.Timestamp("2020-08-31"), "2020-01", tmp_path, Gappy(), lambda s: None, lambda m: None)
    assert out == {"fetched": 3, "missing": 5}


def test_load_round_trip(tmp_path):
    run(tmp_path, Site())
    data = hd.load(tmp_path)
    assert list(data) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28"), pd.Timestamp("2020-03-31")]
    assert data[pd.Timestamp("2020-02-28")].loc["T5", "price"] == 10.0


@pytest.mark.parametrize("a,b", [("RY", "RY.TO"), ("TECK.B", "TECK-B.TO"), ("REI.UN", "REI-UN.TO"), ("BAM/A", "BAM-A.TO")])
def test_yahoo_symbol(a, b):
    assert hd.yahoo_symbol(a) == b
