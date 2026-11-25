"""Swing rank -- company identity, file returns and Yahoo matching. Synthetic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.swing_rank import members as mb
from research.swing_rank.members import (
    accepted, agreement, candidate_symbols, collapse, company_ids, file_returns, link_renames, long_table, match_symbols,
    month_end_returns, split_factor,
)

D = [pd.Timestamp(x) for x in pd.date_range("2020-01-31", periods=30, freq="BME")]


def table(*rows):
    """rows: (file number, ticker, name[, price, shares])"""
    out = [{"date": D[r[0]], "ticker": r[1], "name": r[2], "price": r[3] if len(r) > 3 else 10.0,
            "shares": r[4] if len(r) > 4 else 1000.0, "sector": "X"} for r in rows]
    t = pd.DataFrame(out).sort_values(["date", "ticker"]).reset_index(drop=True)
    t["id"] = company_ids(t)
    return t


def ids(t):
    return {(a, b): c for a, b, c in zip(t["ticker"], t["name"], t["id"])}


def test_long_table():
    f = {D[0]: pd.DataFrame({"name": ["A CO"], "price": [1.0]}, index=pd.Index(["A"], name="ticker")),
         D[1]: pd.DataFrame({"name": ["A CO", "B CO"], "price": [1.0, 2.0]}, index=pd.Index(["A", "B"], name="ticker"))}
    t = long_table(f)
    assert t[["date", "ticker"]].values.tolist() == [[D[0], "A"], [D[1], "A"], [D[1], "B"]]


def test_same_ticker_new_name_is_one_company():
    t = table((0, "SU", "SUNCOR"), (1, "SU", "SUNCOR"), (2, "SU", "SUNCOR ENERGY INC"))
    assert t["id"].nunique() == 1


def test_same_name_new_ticker_is_one_company():
    t = table((0, "BBRY", "BLACKBERRY LTD"), (1, "BB", "BLACKBERRY LTD"))
    assert t["id"].nunique() == 1


def test_chain_of_changes():
    t = table((0, "RIM", "RESEARCH IN MOTION"), (1, "BBRY", "RESEARCH IN MOTION"), (2, "BBRY", "BLACKBERRY LTD"), (3, "BB", "BLACKBERRY LTD"))
    assert t["id"].nunique() == 1


def test_one_ticker_two_companies_in_the_same_file():
    t = table((0, "IAG", "IAMGOLD CORP"), (0, "IAG", "INDUSTRIAL ALLIANCE", 33.0), (1, "IAG", "IAMGOLD CORP"), (1, "IAG", "INDUSTRIAL ALLIANCE", 33.0))
    i = ids(t)
    assert i[("IAG", "IAMGOLD CORP")] != i[("IAG", "INDUSTRIAL ALLIANCE")]


def test_two_companies_never_merge_through_a_chain():
    t = table((0, "A", "ALPHA"), (0, "B", "BETA", 33.0), (1, "A", "ALPHA"), (1, "B", "BETA", 33.0), (2, "A", "BETA", 33.0), (3, "A", "BETA", 33.0))
    i = ids(t)
    assert i[("A", "ALPHA")] != i[("B", "BETA")]


def test_a_merged_company_remembers_all_its_dates():
    t = table((1, "B", "GAMMA"), (2, "B", "ALPHA"), (2, "A", "ALPHA", 33.0))
    i = ids(t)
    assert i[("B", "GAMMA")] == i[("B", "ALPHA")] != i[("A", "ALPHA")]


def test_a_short_lived_label_does_not_hide_the_long_running_one():
    """The third label starts right after the first one ends; the second, gone
    for two years, started in between."""
    rows = [(k, "CLS", "CELESTICA INC") for k in range(28) if k != 3]
    t = table(*rows, (3, "CLS", "CELESTICA INC."), (28, "CLS", "CELESTICA"), (29, "CLS", "CELESTICA"))
    assert t["id"].nunique() == 1


def test_a_new_label_continues_the_one_seen_last():
    rows = [(k, "X", "FIRST") for k in range(3)] + [(k, "X", "SECOND") for k in range(3, 28)]
    assert table(*rows, (28, "X", "THIRD"))["id"].nunique() == 1


def test_a_new_label_does_not_continue_another_company_of_that_ticker():
    rows = [(k, "X", "ALPHA") for k in range(28)]
    t = table(*rows, (3, "X", "BETA", 33.0), (28, "X", "ALPHA NEW"))
    i = ids(t)
    assert i[("X", "ALPHA NEW")] == i[("X", "ALPHA")] != i[("X", "BETA")]


def test_ticker_reused_after_a_long_absence_is_another_company():
    t = table((0, "ABC", "OLD CO"), (20, "ABC", "NEW CO"))
    assert t["id"].nunique() == 2
    t = table((0, "ABC", "OLD CO"), (12, "ABC", "NEW CO"))
    assert t["id"].nunique() == 1


def test_company_listed_twice_in_one_file_is_one_company():
    """December 2011: ARC under AETUF and ARX, same name and price, split share count."""
    t = table((0, "AETUF", "ARC RESOURCES", 25.1, 700), (1, "AETUF", "ARC RESOURCES", 25.0, 300), (1, "ARX", "ARC RESOURCES", 25.0, 400),
              (2, "ARX", "ARC RESOURCES", 24.0, 700))
    assert t["id"].nunique() == 1
    c = collapse(t)
    assert len(c) == 3 and c["shares"].tolist() == [700, 700, 700] and c["price"].tolist() == [25.1, 25.0, 24.0]


def test_collapse_leaves_single_rows_alone():
    t = table((0, "A", "A", 10.0, 5), (0, "B", "B", 11.0, 6), (1, "A", "A", 12.0, 7))
    pd.testing.assert_frame_equal(collapse(t), t)


def test_prices_half_a_percent_apart_are_two_securities():
    assert table((0, "A", "SAME", 10.0), (0, "B", "SAME", 10.06))["id"].nunique() == 2
    assert table((0, "A", "SAME", 10.0), (0, "B", "SAME", 10.04))["id"].nunique() == 1


def test_share_classes_stay_apart():
    t = table((0, "TECK.A", "TECK", 40.0), (0, "TECK.B", "TECK", 39.0), (1, "TECK.A", "TECK", 41.0), (1, "TECK.B", "TECK", 40.0))
    assert t["id"].nunique() == 2


# ── splits and file returns ──────────────────────────────────────────────────

@pytest.mark.parametrize("p,s,f", [
    (0.50, 2.0, 2.0), (0.34, 3.0, 3.0), (10.2, 0.1, 0.1), (0.66, 1.5, 1.5), (0.55, 2.02, 2.0),
    (0.50, 1.0, 1.0),                    # halved without more shares: a real fall
    (1.05, 2.0, 1.0),                    # the fund doubled its position
    (0.95, 1.5, 1.0), (1.0, 1.0, 1.0), (2.1, 1.0, 1.0),
    (0.20, 2.0, 1.0), (0.95, 2.0, 1.0),  # far more than a split explains
    (24.09 / 132.46, 5.7, 5.0),          # First Quantum 2011: 5-for-1, the fund also bought
    (19.27 / 1.02, 0.08, 1 / 12),        # Aurora 2020: 1-for-12 in a +57% month
    (19.32 / 1.19, 0.04, 1 / 25)])       # Bombardier 2022: 1-for-25 in a -35% month
def test_split_factor(p, s, f):
    assert split_factor(p, s) == f


def test_file_returns_by_hand():
    t = table((0, "A", "A", 10.0), (1, "A", "A", 11.0), (2, "A", "A", 5.6, 2000.0), (3, "A", "A", 5.0, 2000.0))
    r = file_returns(t)[t["id"].iloc[0]]
    assert r[D[1]] == pytest.approx(0.10) and r[D[2]] == pytest.approx(5.6 * 2 / 11 - 1) and r[D[3]] == pytest.approx(5 / 5.6 - 1)


def test_file_returns_skip_gaps():
    t = table((0, "A", "A"), (1, "A", "A"), (1, "B", "B"), (2, "B", "B"), (3, "A", "A"), (3, "B", "B"))
    r = file_returns(t)
    i = ids(t)
    assert r[i[("A", "A")]].dropna().index.tolist() == [D[1]]
    assert r[i[("B", "B")]].dropna().index.tolist() == [D[2], D[3]]


def test_file_returns_skip_months_the_publisher_missed():
    t = table((0, "A", "A"), (3, "A", "A"))             # consecutive files, three months apart
    assert file_returns(t).empty or file_returns(t).isna().all().all()


# ── renames ──────────────────────────────────────────────────────────────────

def renamed(*rows):
    t = table(*rows)
    t["id"] = link_renames(t)
    return t


def test_holding_that_changes_ticker_and_name_is_one_company():
    t = renamed((0, "ECA", "ENCANA", 10.0, 12345), (0, "X", "X"), (1, "OVV", "OVINTIV", 10.5, 12400), (1, "X", "X"), (2, "OVV", "OVINTIV", 11.0, 12400))
    assert t.loc[t["ticker"].isin(["ECA", "OVV"]), "id"].nunique() == 1 and t["id"].nunique() == 2


def test_chain_of_renames():
    t = renamed((0, "A", "A", 10.0, 500), (1, "B", "B", 10.0, 500), (2, "C", "C", 10.0, 500))
    assert t["id"].nunique() == 1


@pytest.mark.parametrize("price,shares", [(12.1, 1000), (7.9, 1000), (10.0, 1031), (10.0, 969)])
def test_rename_needs_the_same_holding(price, shares):
    assert renamed((0, "A", "A", 10.0, 1000), (1, "B", "B", price, shares))["id"].nunique() == 2


def test_rename_must_be_one_to_one():
    t = renamed((0, "A", "A", 10.0, 1000), (1, "B", "B", 10.0, 1000), (1, "C", "C", 10.0, 1000))
    assert t["id"].nunique() == 3
    t = renamed((0, "A", "A", 10.0, 1000), (0, "B", "B", 10.0, 1000), (1, "C", "C", 10.0, 1000))
    assert t["id"].nunique() == 3


def test_rename_only_between_one_that_left_and_one_that_is_new():
    t = renamed((0, "A", "A", 10.0, 1000), (1, "A", "A", 10.0, 1000), (1, "B", "B", 10.0, 1000))
    assert t["id"].nunique() == 2                        # A is still there
    t = renamed((0, "A", "A", 10.0, 1000), (0, "B", "B", 50.0, 77), (1, "B", "B", 10.0, 1000))
    assert t["id"].nunique() == 2                        # B is not new


def test_rename_needs_the_same_sector():
    t = table((0, "A", "A", 10.0, 1000), (1, "B", "B", 10.0, 1000))
    t["sector"] = ["Energy", "Utilities"]
    assert link_renames(t).nunique() == 2


# ── Yahoo ────────────────────────────────────────────────────────────────────

def test_candidate_symbols():
    t = table((0, "RY", "a"), (0, "TECK.B", "b"), (0, "945", "c"), (0, "0P6E", "d"), (0, "REI.UN", "e"), (0, "FET UN", "f"), (0, "AL.1", "g"))
    assert candidate_symbols(t) == ["FET-UN.TO", "REI-UN.TO", "RY.TO", "TECK-B.TO"]


def test_month_end_returns():
    idx = pd.bdate_range("2020-01-01", "2020-08-31")
    c = pd.DataFrame({"A": np.arange(len(idx)) + 100.0}, index=idx)
    c.loc[D[2], "A"] = np.nan                            # no trade that day: the previous close counts
    r = month_end_returns(c, [D[0], D[1], D[2], D[5]])
    assert r["A"].iloc[1] == pytest.approx(c["A"][D[1]] / c["A"][D[0]] - 1)
    assert r["A"].iloc[2] == pytest.approx(c["A"].iloc[idx.get_loc(D[2]) - 1] / c["A"][D[1]] - 1)
    assert np.isnan(r["A"].iloc[0]) and np.isnan(r["A"].iloc[3])     # three months apart


def series(seed, n=24, start=0):
    return pd.Series(np.random.default_rng(seed).normal(0.005, 0.08, n), index=D[start:start + n])


def test_accepts_dividends_but_not_another_stock():
    f = series(1)
    assert accepted(f, f + 0.003) and accepted(f, f)
    assert not accepted(f, series(2)) and not accepted(f, f + 0.02)


def test_acceptance_needs_six_months_and_correlation():
    f = series(1)
    assert accepted(f.iloc[:6], f) and not accepted(f.iloc[:5], f)
    g = f * 0.0 + np.random.default_rng(0).normal(0, 0.004, len(f))
    h = g + np.random.default_rng(5).normal(0, 0.004, len(f))
    assert agreement(g, h)[1] < 0.01 and not accepted(g, h)          # both flat: close but unrelated


def setup_match():
    rows = []
    prices = {}
    rng = np.random.default_rng(3)
    for k, (t, n) in enumerate([("AAA", "ALPHA"), ("OLD", "RENAMED"), ("ZZZ", "WRONG"), ("GONE", "DEAD")]):
        p = 10 * np.exp(np.cumsum(rng.normal(0, 0.08, 24)))
        prices[t] = p
        rows += [(j, t, n, p[j]) for j in range(24)]
    t = table(*rows)
    fr = file_returns(t)
    r = lambda p: pd.Series(p, index=D[:24]).pct_change()
    yahoo = pd.DataFrame({"AAA.TO": r(prices["AAA"]), "NEW.TO": r(prices["OLD"]) + 0.001,
                          "ZZZ.TO": series(77), "OTHER.TO": series(78)})
    return t, fr, yahoo


def test_match_own_ticker_renamed_and_unmatched():
    t, fr, yahoo = setup_match()
    i = {a: c for (a, _), c in ids(t).items()}
    m = match_symbols(t, fr, yahoo)
    assert m == {i["AAA"]: "AAA.TO", i["OLD"]: "NEW.TO"}


def test_match_symbol_is_used_once():
    t, fr, yahoo = setup_match()
    i = {a: c for (a, _), c in ids(t).items()}
    fr[i["GONE"]] = fr[i["AAA"]]                         # a twin that would also fit AAA.TO
    assert list(match_symbols(t, fr, yahoo).values()).count("AAA.TO") == 1


def test_own_symbol_is_also_used_once():
    p = 10 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.08, 24)))
    t = table(*[(j, "AAA", "FIRST", p[j]) for j in range(12)], *[(j, "AAA", "SECOND", p[j - 4]) for j in range(4, 16)])
    assert t["id"].nunique() == 2
    fr = file_returns(t)
    i = ids(t)
    fr[i[("AAA", "SECOND")]] = fr[i[("AAA", "FIRST")]]
    m = match_symbols(t, fr, pd.DataFrame({"AAA.TO": pd.Series(p, index=D[:24]).pct_change()}))
    assert list(m.values()) == ["AAA.TO"]


def test_match_prefers_the_newest_own_ticker():
    p = 10 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.08, 24)))
    t = table(*[(j, "OLD" if j < 12 else "NEW", "SAME CO", p[j]) for j in range(24)])
    fr = file_returns(t)
    r = pd.Series(p, index=D[:24]).pct_change()
    assert list(match_symbols(t, fr, pd.DataFrame({"OLD.TO": r, "NEW.TO": r})).values()) == ["NEW.TO"]


def test_search_by_returns_needs_a_year():
    p = 10 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.08, 10)))
    t = table(*[(j, "OLD", "SAME CO", p[j]) for j in range(10)])
    fr = file_returns(t)
    r = pd.Series(p, index=D[:10]).pct_change()
    assert match_symbols(t, fr, pd.DataFrame({"OTHER.TO": r})) == {}
    assert list(match_symbols(t, fr, pd.DataFrame({"OLD.TO": r})).values()) == ["OLD.TO"]
