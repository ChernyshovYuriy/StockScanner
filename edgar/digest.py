"""
EDGAR digest builder — assembles the plain-text email of FLAGGED HITS ONLY.

Interim flagging (Step 2 deferred): watchlist insider purchases (Form 4 code
'P' — SEC's own table defines this as open-market OR privately-negotiated;
the data alone can't tell the two apart, so a single large 'P' buy could be
a private block trade rather than one made on the tape) + all SC 13D/13G
activist/passive stakes market-wide. A quiet day returns None, so the
service sends nothing.

Honest ceiling: every filing is a LAGGED disclosure (4-10+ days). This surfaces
*footprints* of conviction capital earlier and more systematically than the
crowd — a research trigger, never a price predictor or financial advice. Sells
carry little signal and are intentionally never flagged.
"""

FOOTER = (
    "research triggers — not recommendations. one filing is never enough to act.\n"
    "every filing is a lagged disclosure (you follow footprints, not live\n"
    "position); sells are intentionally never flagged. not financial advice."
)

# Daily-index form string -> short label for the digest (both spellings).
_ACTIVIST_FORMS = {
    "SC 13D": "13D", "SC 13D/A": "13D/A",
    "SC 13G": "13G", "SC 13G/A": "13G/A",
    "SCHEDULE 13D": "13D", "SCHEDULE 13D/A": "13D/A",
    "SCHEDULE 13G": "13G", "SCHEDULE 13G/A": "13G/A",
}


def build_digest(digest_date, insider_buys, activist_hits, new_insiders=None):
    """
    Build (subject, body) for the day's flagged hits, or None on a quiet day.

    insider_buys : dicts with ticker, owner, shares, price, date (txn date)
    activist_hits: scan-hit dicts with ticker, cik, form, date, url
    new_insiders : scan-hit dicts (Form 3 -- new Section 16 filer) with
                   ticker, cik, date, url. Optional; omit or [] if unused.
    """
    new_insiders = new_insiders or []
    if not insider_buys and not activist_hits and not new_insiders:
        return None

    lines = [f"EDGAR digest — {digest_date}", ""]

    if insider_buys:
        lines.append("INSIDER BUYS (watchlist, code P: open-market or private purchase)")
        # (ticker, owner) -> [lot count, total shares, total value], so
        # multiple same-day lots by one insider get one rollup line after
        # their individual entries instead of only being readable one by one.
        lot_totals = {}
        for b in insider_buys:
            ticker = b.get("ticker") or f"CIK{b.get('cik', '?')}"
            owner = (b.get("owner") or "?")[:24]
            shares = b.get("shares") or 0
            price = b.get("price")
            price_s = f"${price:,.2f}" if price is not None else "$?"
            # 2+ distinct insiders buying the same name the same day -- the
            # strongest form of this signal -- gets called out on the line.
            cluster_s = "   >> CLUSTER" if b.get("cluster") else ""
            lines.append(
                f"  {ticker:<8} {owner:<24} {shares:>10,.0f} sh @ {price_s}   "
                f"{b.get('date', '')}{cluster_s}"
            )
            tot = lot_totals.setdefault((ticker, owner), [0, 0.0, 0.0])
            tot[0] += 1
            tot[1] += shares
            tot[2] += shares * (price or 0)
        for (ticker, owner), (n, tot_shares, tot_value) in lot_totals.items():
            if n > 1:
                lines.append(
                    f"    -> {owner} ({ticker}): {n} lots, {tot_shares:,.0f} sh total, "
                    f"${tot_value:,.0f}"
                )
        lines.append("")

    if new_insiders:
        lines.append("NEW INSIDERS (Form 3 -- just became subject to Section 16)")
        for h in new_insiders:
            ticker = h.get("ticker") or f"CIK{h.get('cik', '?')}"
            lines.append(f"  {ticker:<8} {h.get('date', '')}")
            if h.get("url"):
                lines.append(f"           {h['url']}")
        lines.append("")

    if activist_hits:
        lines.append("ACTIVIST STAKES (13D / 13G)")
        for h in activist_hits:
            ticker = h.get("ticker") or f"CIK{h.get('cik', '?')}"
            form = _ACTIVIST_FORMS.get(h.get("form"), h.get("form", ""))
            bits = [f"  {ticker:<8} {form:<6} {h.get('date', '')}"]
            if h.get("filer"):
                bits.append(f"by {h['filer']}")
            if h.get("pct") is not None:
                bits.append(f"{h['pct']:.1f}%")
            lines.append("   ".join(bits))
            if h.get("url"):
                lines.append(f"           {h['url']}")
        lines.append("")

    lines.append(FOOTER)
    body = "\n".join(lines)

    n_ins, n_act, n_new = len(insider_buys), len(activist_hits), len(new_insiders)
    subject = (
        f"EDGAR: {n_ins} insider buy{'' if n_ins == 1 else 's'}, {n_act} activist"
    )
    if n_new:
        subject += f", {n_new} new insider{'' if n_new == 1 else 's'}"
    return subject, body
