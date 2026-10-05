"""
press_release_tracker/financing.py
====================================
Structured financing terms for a financing press release (added 2026-10).
analyst.py already describes dilution in prose plus a coarse
none/minor/significant label; that's readable but not testable. This pulls
the TERMS out as plain fields (offering type, proceeds, issue price,
securities offered, warrants, flow-through, brokered, insider/strategic
participation, deal stage) so news_watchlist/outcomes.py can bucket
releases by discount-to-market and dilution and check which kinds of deals
actually underperform.

The LLM only EXTRACTS what the text states -- it does no arithmetic. The
two derived numbers are computed in code by news_watchlist/outcomes.py:
discount = issue price vs the last close before publication (from daily
bars), dilution = securities offered / shares outstanding (Yahoo, captured
here at extraction time -- for a backfilled release that's TODAY's count,
which may already include the deal's own shares, so dilution reads a bit
low for those).

Runs live from press_release_service.py for every English release the
classifier tags 'financing', reusing the article body the analyst read
already fetched. `python -m press_release_tracker.financing --backfill`
fills in releases parsed before this existed -- safe from lookahead,
because the terms are written in the release text itself, unlike a
verdict, which the model could colour with what it knows happened next.

OPENAI_API_KEY is shared with llm_parser.py; unconfigured / failed fetch /
failed call -> None, nothing stored, same "degrade gracefully" convention.
"""
from __future__ import annotations

import argparse
import json
from typing import Optional

from press_release_tracker import analyst, article
from press_release_tracker.llm_parser import OPENAI_API_KEY

MODEL = analyst.MODEL
MAX_BODY_CHARS = analyst.MAX_BODY_CHARS

_SYSTEM_PROMPT = (
    "You extract the terms of a securities financing from the full text of one company "
    "press release (small Canadian listed companies: TSX, TSX Venture, CSE). Extract only "
    "what the text states; never compute or guess -- use null for anything not stated.\n"
    "Reply with ONLY a JSON object with these keys:\n"
    "is_financing: true if the release announces, amends, prices or closes an equity, unit, "
    "convertible or debt financing of this company; false otherwise (then the rest may be null).\n"
    "deal_stage: one of announced, upsized, amended, priced, closed, other. A release that "
    "both announces and closes counts as closed.\n"
    "offering_type: one of private_placement, life_offering (listed issuer financing "
    "exemption), bought_deal, public_offering, rights_offering, convertible, debt, "
    "strategic_investment, other.\n"
    "brokered: true, false, or null.\n"
    "flow_through: true if any part is flow-through shares, else false.\n"
    "gross_proceeds: total gross proceeds as a number in units of currency (e.g. 2500000), "
    "the maximum stated including any over-allotment only if it says so; null if not stated.\n"
    "currency: ISO code of the proceeds/price currency (CAD, USD, ...); C$ or $ for a "
    "Canadian issuer with no other indication is CAD.\n"
    "issue_price: price per share or unit as a number. If there are several tranches at "
    "different prices, give the non-flow-through (hard-dollar) price.\n"
    "securities_offered: number of shares or units offered/issued as a number, the maximum "
    "stated; null if not stated.\n"
    "warrant_coverage: warrants per unit (e.g. 1.0 for one full warrant, 0.5 for one half); "
    "0 if the securities carry no warrants.\n"
    "warrant_strike: warrant exercise price as a number, or null.\n"
    "warrant_term_months: warrant term in months, or null.\n"
    "insider_participation: true if directors/officers/insiders are stated to participate, "
    "else false.\n"
    "strategic_investor: name of a named strategic or lead investor, or null.\n"
    "use_of_proceeds: at most 12 words, or null."
)

_FIELDS = (
    "is_financing", "deal_stage", "offering_type", "brokered", "flow_through",
    "gross_proceeds", "currency", "issue_price", "securities_offered",
    "warrant_coverage", "warrant_strike", "warrant_term_months",
    "insider_participation", "strategic_investor", "use_of_proceeds",
)


def should_extract(parsed: Optional[dict], link: str) -> bool:
    """A parsed ticker, an English page (same rule as analyst.py), and the
    classifier's category 'financing'."""
    if not parsed or not parsed.get("ticker") or parsed.get("category") != "financing":
        return False
    m = analyst._LANG_RE.search(link or "")
    return not (m and m.group(1) != "en")


def _number(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def normalize(raw: dict) -> dict:
    """Keep the known keys, coerce numbers (the model sometimes returns
    "2,500,000" as a string)."""
    out = {k: raw.get(k) for k in _FIELDS}
    for k in ("gross_proceeds", "issue_price", "securities_offered", "warrant_coverage",
              "warrant_strike", "warrant_term_months"):
        out[k] = _number(out[k])
    if out["currency"]:
        out["currency"] = str(out["currency"]).upper().replace("C$", "CAD").strip()
    return out


def extract_terms(ticker: str, title: str, link: str, body: Optional[str] = None,
                  context: Optional[dict] = None) -> Optional[dict]:
    """Fetch the article (unless body is given) and Yahoo context (unless
    given), run one extraction call. Returns normalize()d terms plus
    shares_outstanding/yahoo_ticker/market_cap from the context, or None."""
    if not OPENAI_API_KEY:
        return None
    body = body if body is not None else article.fetch_article_text(link)
    if not body:
        return None
    context = context if context is not None else analyst.market_context(ticker)

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Ticker: {ticker}\nTitle: {title}\n\n"
                                            f"Full release text:\n{body[:MAX_BODY_CHARS]}"},
            ],
            response_format={"type": "json_object"},
            timeout=180,
        )
        raw = json.loads(resp.choices[0].message.content)
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    terms = normalize(raw)
    ctx = context or {}
    terms["yahoo_ticker"] = ctx.get("yahoo_ticker")
    terms["shares_outstanding"] = _number(ctx.get("shares_outstanding"))
    terms["market_cap"] = _number(ctx.get("market_cap"))
    return terms


# ── backfill ─────────────────────────────────────────────────────────────

def _backfill(limit: Optional[int], dry_run: bool, workers: int) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from press_release_tracker import store
    from time_utils import market_now

    conn = store.connect()
    todo = [r for r in store.financing_without_terms(conn)
            if should_extract({"ticker": r["ticker"], "category": "financing"}, r["link"])]
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} financing release(s) to extract")

    def work(r):
        return r, extract_terms(r["ticker"], r["title"], r["link"])

    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (r, terms) in enumerate(pool.map(work, todo), 1):
            if terms is None:
                print(f"[{i}/{len(todo)}] {r['ticker']}: FAILED")
                continue
            ok += 1
            print(f"[{i}/{len(todo)}] {r['ticker']}: {terms['deal_stage']} {terms['offering_type']} "
                  f"{terms['gross_proceeds']} {terms['currency']} @ {terms['issue_price']} "
                  f"x{terms['securities_offered']} warrants={terms['warrant_coverage']} "
                  f"FT={terms['flow_through']}")
            if not dry_run:
                store.save_financing_terms(conn, r["guid"], r["ticker"], terms, MODEL,
                                           market_now().isoformat())
    print(f"done: {ok}/{len(todo)} extracted" + (" (dry run, nothing saved)" if dry_run else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Financing-terms extraction (see module docstring)")
    p.add_argument("--backfill", action="store_true",
                   help="Extract terms for every parsed financing release that has none yet.")
    p.add_argument("--limit", type=int, default=None, help="Only the first N (testing).")
    p.add_argument("--workers", type=int, default=4, help="Parallel fetch+LLM calls.")
    p.add_argument("--dry-run", action="store_true", help="Print, save nothing.")
    args = p.parse_args()
    if not args.backfill:
        p.error("nothing to do -- pass --backfill")
    _backfill(args.limit, args.dry_run, args.workers)
