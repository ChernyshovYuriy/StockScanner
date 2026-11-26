"""US list: which securities count, and the files as saved. No network."""
from __future__ import annotations

import pandas as pd
import pytest

from research.us import universe as u


@pytest.mark.parametrize("symbol,name,want", [
    ("AAPL", "Apple Inc. - Common Stock", True), ("BRK.B", "Berkshire Hathaway Inc. New Common Stock", True), ("GOOGL", "Alphabet Inc. - Class A Common Stock", True),
    ("BABA", "Alibaba Group Holding Limited American Depositary Shares, each representing eight Ordinary Shares", True),
    ("SHOP", "Shopify Inc. - Class A subordinate voting shares", True), ("EPD", "Enterprise Products Partners L.P. Common Units", False),
    ("JPM$D", "JPMorgan Chase & Co. Depositary Shares, each representing a 1/400th interest", False), ("ABCW", "ABC Corp. - Warrant", False),
    ("ABCU", "ABC Acquisition Corp. - Units", False), ("ABC", "ABC Acquisition Corp. - Class A Ordinary Shares", False), ("XYZ", "XYZ Municipal Fund Common Stock", False),
    ("QQQZ", "Foo 5.5% Senior Notes due 2030", False), ("PREF", "Bank 6% Perpetual Preferred Common Stock", False), ("ABCR", "ABC Inc. - Right", False), ("ZZ", "Zed Index", False)])
def test_is_common(symbol, name, want):
    assert u.is_common(symbol, name) is want


def test_build_drops_etfs_tests_and_other_exchanges(tmp_path):
    (tmp_path / "n").write_text("Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
                                "AAA|A Inc. - Common Stock|Q|N|N|100|N|N\nTST|T Inc. - Common Stock|Q|Y|N|100|N|N\nETF|E - Common Stock|Q|N|N|100|Y|N\nFile Creation Time: 1|||||||\n")
    (tmp_path / "o").write_text("ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
                                "BF.B|B Inc. Class B Common Stock|N|x|N|100|N|x\nARC|C Common Stock|P|x|N|100|N|x\nAMX|D Common Stock|A|x|N|100|N|x\nNA|Nano Labs Common Stock|N|x|N|100|N|x\nFile Creation Time: 1|||||||\n")
    t = u.build(tmp_path / "n", tmp_path / "o")
    assert t["ticker"].tolist() == ["AAA", "AMX", "BF-B", "NA"] and t["exchange"].tolist() == ["Nasdaq", "NYSE American", "NYSE", "NYSE"]


def test_members_keeps_spells_after_the_start(tmp_path):
    (tmp_path / "m").write_text("ticker,start_date,end_date\nOLD,1999-01-01,2011-10-31\nA,2000-06-05,\nBF.B,2000-01-01,2015-01-01\n")
    assert u.members(tmp_path / "m").to_numpy().tolist() == [["A", "2000-06-05", ""], ["BF-B", "2000-01-01", "2015-01-01"]]


def test_saved_lists():
    t = u.load_tickers()
    assert 3500 < len(t) < 7000 and {"AAPL", "MSFT", "BRK-B", "TGTX", "KAI"} <= set(t) and "SPY" not in t and len(t) == len(set(t))
    m = pd.read_csv(u.SP500, keep_default_na=False)
    n = ((m["start"] <= "2020-01-02") & ((m["end"] == "") | (m["end"] > "2020-01-02"))).sum()
    assert 495 <= n <= 510
