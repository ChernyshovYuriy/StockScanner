"""
Digest builder -- assembles the daily plain-text email. Unlike
edgar/digest.py (quiet day = None, no email), this ALWAYS returns a
(subject, body): a paper-tracking experiment is meant to be watched every
day, even a day with no new activity, so the open-position list itself is
the report on a quiet day.
"""

FOOTER = (
    "research tool -- tracks what price does after a Triple Screen BUY, not a\n"
    "trade recommendation. exit rule is a zero-tolerance close < buy_price,\n"
    "not a simulated stop; review the price history to judge a real exit point."
)


def _pct(buy_price, other_price):
    if not buy_price:
        return None
    return (other_price / buy_price - 1.0) * 100.0


def build_digest(digest_date, new_buys, new_sells, open_rows):
    """
    Build (subject, body) for the day's activity plus the full open list.

    new_buys : dicts with ticker, buy_price, trend, pullback, trigger
    new_sells: dicts with ticker, buy_date, buy_price, sell_date, sell_price
    open_rows: dicts with ticker, buy_date, buy_price, latest_price, days_held
    """
    lines = [f"Triple Screen Tracker — {digest_date}", ""]

    if new_buys:
        lines.append("NEW BUYS")
        for b in new_buys:
            lines.append(
                f"  {b['ticker']:<8} bought @ ${b['buy_price']:,.2f}   "
                f"trend={b['trend']} pullback={b['pullback']} trigger={b['trigger']}"
            )
        lines.append("")

    if new_sells:
        lines.append("SOLD (closed below entry)")
        for s in new_sells:
            pct = _pct(s["buy_price"], s["sell_price"])
            pct_s = f"{pct:+.1f}%" if pct is not None else "—"
            lines.append(
                f"  {s['ticker']:<8} {s['buy_date']} @ ${s['buy_price']:,.2f}  ->  "
                f"{s['sell_date']} @ ${s['sell_price']:,.2f}   ({pct_s})"
            )
        lines.append("")

    lines.append(f"OPEN POSITIONS ({len(open_rows)})")
    if not open_rows:
        lines.append("  none")
    else:
        for r in open_rows:
            pct = _pct(r["buy_price"], r["latest_price"])
            pct_s = f"{pct:+.1f}%" if pct is not None else "—"
            lines.append(
                f"  {r['ticker']:<8} bought {r['buy_date']} @ ${r['buy_price']:,.2f}   "
                f"now ${r['latest_price']:,.2f} ({pct_s}, day {r['days_held']})"
            )
    lines.append("")

    lines.append(FOOTER)
    body = "\n".join(lines)

    n_buy, n_sell, n_open = len(new_buys), len(new_sells), len(open_rows)
    subject = f"Triple Screen Tracker {digest_date}: {n_buy} new buy{'' if n_buy == 1 else 's'}, {n_sell} sold, {n_open} open"
    return subject, body
