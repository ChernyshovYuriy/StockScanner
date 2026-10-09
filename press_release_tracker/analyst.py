"""
press_release_tracker/analyst.py
==================================
The "analyst read" of one important press release (added 2026-10): where
llm_parser.py classifies an item from its short RSS teaser, this reads the
FULL article (article.py) and pulls out what a careful investor needs --
the actual figures and their comparisons, business trend, cash runway,
dilution (issue price vs the current price, shares added as % of shares
outstanding), red flags, and a verdict. The point is reading what anyone
COULD read but almost nobody does: most TSXV/CSE names have no analyst
coverage, so their quarterly results and financings go unread.

Only items worth the cost are analysed (should_analyze(): a ticker, an
English-language page, and a category in
config.PRESS_RELEASE_ANALYSIS_CATEGORIES or materiality 'high'). Market
context (price, market cap, shares outstanding) comes from Yahoo via
LiveDataProvider.get_info() -- best effort; a junior with no Yahoo data
is still analysed, just without the dilution arithmetic.

OPENAI_API_KEY is shared with llm_parser.py. Unconfigured, a failed fetch,
or a failed/malformed LLM call -> analyze_release() returns None and the
item goes out exactly as before, same "degrade gracefully" convention.

The verdicts are UNVALIDATED. News Watchlist's price history is what can
score them later, the same way every other signal in this repo was tested.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from press_release_tracker import article
from press_release_tracker.llm_parser import OPENAI_API_KEY

try:
    from config import PRESS_RELEASE_ANALYSIS_CATEGORIES as CATEGORIES
    from config import PRESS_RELEASE_ANALYSIS_MAX_BODY_CHARS as MAX_BODY_CHARS
    from config import PRESS_RELEASE_ANALYSIS_MODEL as MODEL
except Exception:
    MODEL = "gpt-5-mini"
    MAX_BODY_CHARS = 24000
    CATEGORIES = ("earnings", "financing", "ma_acquisition", "contract_award")

# GlobeNewswire repeats one release in several languages
# (".../3373331/0/en/...", ".../0/fr/...", ".../0/de/...") -- analyse the
# English copy only.
_LANG_RE = re.compile(r"/\d+/\d+/([a-z]{2})/")

_SYSTEM_PROMPT = (
    "You are a skeptical equity analyst covering small Canadian listed companies "
    "(TSX, TSX Venture, CSE) that few analysts follow. You read the full text of one "
    "company press release and report what a careful investor needs to know. Work only "
    "from the text and the market context given; never invent numbers -- if a figure "
    "isn't there, use null. Keep the currency the release uses (many report in US$). "
    "Judge the substance, not the release's promotional tone.\n"
    "Reply with ONLY a JSON object with these keys:\n"
    "release_type: one of financial_results, results_date_notice, financing, "
    "ma_acquisition, contract_award, operational_update, other.\n"
    "period: the reporting period covered (e.g. \"Q3 2026\", \"FY ended 2026-06-30\") or null.\n"
    "key_figures: list of up to 8 short strings, one figure each with its comparison where "
    "given, e.g. \"Revenue US$165.7M vs 98.6M FY2025 (+68%)\".\n"
    "what_is_new: 1-2 sentences: the most important new information, including anything "
    "material the headline doesn't say (buried in the body or the tables).\n"
    "trend: one of improving, stable, deteriorating, unclear -- the direction of the "
    "underlying business per the figures (revenue, margins, profit or loss, cash flow).\n"
    "cash_position: one sentence on cash / working capital and, for a loss-making company, "
    "roughly how many quarters of runway at the stated burn; null if not given.\n"
    "dilution: object {level: one of none, minor, significant, unknown; detail: one "
    "sentence}. New shares, units, warrants or convertibles count; debt and non-dilutive "
    "funding are none. When market context is given, compare the issue price with the "
    "current price and estimate shares added as % of shares outstanding, showing the "
    "arithmetic briefly. Over ~10% of outstanding is significant. Flow-through shares "
    "normally price at a premium to market for tax reasons, so that premium alone is not a "
    "positive.\n"
    "valuation: one or two sentences relating the figures to the market cap when market "
    "context is given -- e.g. P/E on annualised earnings, cash as % of market cap, "
    "price/sales -- converting currencies where needed and stating the rate you assumed; "
    "null if not computable.\n"
    "red_flags: list of short strings, e.g. going-concern language, financing priced below "
    "market, warrants attached, related-party terms, share consolidation, guidance cut, "
    "auditor change, one-off gains flattering profit; [] if none.\n"
    "positives: list of short strings; [] if none.\n"
    "verdict: one of bullish, neutral, bearish -- for a holder of the stock over the next "
    "1-6 months.\n"
    "verdict_reason: one sentence.\n"
    "confidence: one of high, medium, low -- how much hard data the release gives to judge."
)


def should_analyze(parsed: Optional[dict], link: str) -> bool:
    """True when an item is worth the full-article fetch + analysis call:
    a parsed ticker, an English page, and a category in CATEGORIES or
    materiality 'high'."""
    if not parsed or not parsed.get("ticker"):
        return False
    m = _LANG_RE.search(link or "")
    if m and m.group(1) != "en":
        return False
    return parsed.get("category") in CATEGORIES or parsed.get("materiality") == "high"


def _candidates(ticker: str) -> list[str]:
    """Same order as news_watchlist_service._resolve_market_price(): a
    bare symbol is tried as TSX (.TO), then TSX Venture (.V), then as-is.
    The parser's ".CSE" suffix is Yahoo's ".CN"."""
    if ticker.upper().endswith(".CSE"):
        ticker = ticker[:-4] + ".CN"
    if "." in ticker:
        return [ticker]
    return [ticker + ".TO", ticker + ".V", ticker]


def market_context(ticker: str) -> Optional[dict]:
    """{yahoo_ticker, price, market_cap, shares_outstanding, currency} from
    the first candidate symbol Yahoo has a price for, or None. Best effort
    -- any Yahoo failure is just "no context"."""
    from market_data import LiveDataProvider

    provider = LiveDataProvider()
    for symbol in _candidates(ticker):
        try:
            info = provider.get_info(symbol) or {}
        except Exception:
            continue
        price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
        if price:
            return {
                "yahoo_ticker": symbol,
                "price": price,
                "market_cap": info.get("marketCap"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "currency": info.get("currency"),
            }
    return None


def _context_lines(ctx: Optional[dict]) -> str:
    if not ctx:
        return "Market context: not available."
    parts = [f"symbol {ctx['yahoo_ticker']}", f"last price {ctx['price']} {ctx.get('currency') or ''}".strip()]
    if ctx.get("market_cap"):
        parts.append(f"market cap {ctx['market_cap'] / 1e6:,.1f}M")
    if ctx.get("shares_outstanding"):
        parts.append(f"shares outstanding {ctx['shares_outstanding'] / 1e6:,.1f}M")
    return "Market context (Yahoo, may lag the release): " + ", ".join(parts) + "."


def analyze_release(ticker: str, company: Optional[str], title: str, link: str,
                    body: Optional[str] = None, context: Optional[dict] = None) -> Optional[dict]:
    """Fetch the article (unless body is given), add market context
    (unless given), and run one analysis call. Returns the parsed JSON
    dict plus "body_chars" and "market_context", or None when the key is
    unset, the page couldn't be read, or the call/JSON failed."""
    if not OPENAI_API_KEY:
        return None
    body = body if body is not None else article.fetch_article_text(link)
    if not body:
        return None
    context = context if context is not None else market_context(ticker)

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    user_content = (
        f"Ticker: {ticker}\nCompany: {company or '(unknown)'}\nTitle: {title}\n"
        f"{_context_lines(context)}\n\nFull release text:\n{body[:MAX_BODY_CHARS]}"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            timeout=180,
        )
        result = json.loads(resp.choices[0].message.content)
    except Exception:
        return None
    if not isinstance(result, dict):
        return None
    result["body_chars"] = len(body)
    result["market_context"] = context
    return result


def format_analysis(a: dict, indent: str = "  ") -> list[str]:
    """Plain-text lines for an email body (see digest.py)."""
    lines = [f"{indent}ANALYST: {str(a.get('verdict', '?')).upper()} "
             f"(trend {a.get('trend', '?')}, confidence {a.get('confidence', '?')}) — {a.get('verdict_reason') or ''}"]
    if a.get("what_is_new"):
        lines.append(f"{indent}new: {a['what_is_new']}")
    for f in (a.get("key_figures") or [])[:8]:
        lines.append(f"{indent}  • {f}")
    if a.get("valuation"):
        lines.append(f"{indent}valuation: {a['valuation']}")
    if a.get("cash_position"):
        lines.append(f"{indent}cash: {a['cash_position']}")
    d = a.get("dilution") or {}
    if isinstance(d, dict) and d.get("level") not in (None, "none"):
        lines.append(f"{indent}dilution ({d.get('level')}): {d.get('detail') or ''}")
    if a.get("red_flags"):
        lines.append(f"{indent}red flags: " + "; ".join(map(str, a["red_flags"])))
    if a.get("positives"):
        lines.append(f"{indent}positives: " + "; ".join(map(str, a["positives"])))
    return lines
