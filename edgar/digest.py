"""
EDGAR digest builder — assembles the plain-text email of FLAGGED HITS ONLY.

Interim flagging (Step 2 deferred): watchlist open-market insider buys + all
SC 13D/13G activist/passive stakes market-wide.  A quiet day returns None, so
the service sends nothing.

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


def build_digest(digest_date, insider_buys, activist_hits):
    """
    Build (subject, body) for the day's flagged hits, or None on a quiet day.

    insider_buys : dicts with ticker, owner, shares, price, date (txn date)
    activist_hits: scan-hit dicts with ticker, cik, form, date, url
    """
    if not insider_buys and not activist_hits:
        return None

    lines = [f"EDGAR digest — {digest_date}", ""]

    if insider_buys:
        lines.append("INSIDER BUYS (watchlist, open-market)")
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

    n_ins, n_act = len(insider_buys), len(activist_hits)
    subject = (
        f"EDGAR: {n_ins} insider buy{'' if n_ins == 1 else 's'}, {n_act} activist"
    )
    return subject, body
