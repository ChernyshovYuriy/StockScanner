"""
rbc_reports/tagger.py
=====================
Step 2 of the RBC research tracker: labels every stored report with an
LLM, from its title ("Company - headline") and its "Our view" teaser --
the only text a digest carries (the PDF itself needs an Insight login).

Labels (each restricted to a fixed vocabulary; anything else is stored as
the field's fallback, so the dashboard's filters only ever see known
values):
  report_type  company / sector / macro / roundup
  action       initiation, upgrade, downgrade, pt_raise, pt_cut,
               results_beat, results_miss, results_inline, preview,
               estimates, event, none
  tone         bullish / neutral / bearish -- RBC's stance on the subject
  rating       RBC's rating as stated in the text (e.g. "Outperform"), or null
  theme        one of THEMES

Several reports go in one call (config.RBC_TAG_BATCH_SIZE), batches run
in parallel (RBC_TAG_WORKERS), and the run commits once. A failed or
malformed batch is just left untagged -- the next run retries it.
OPENAI_API_KEY is read from .env here, same self-contained pattern as
press_release_tracker/llm_parser.py; unset -> tag_untagged() does nothing.

  python -m rbc_reports.tagger               # tag every untagged report
  python -m rbc_reports.tagger --limit 40    # only the newest 40
  python -m rbc_reports.tagger --retag       # re-tag everything (prompt change)
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import RBC_TAG_BATCH_SIZE, RBC_TAG_MODEL, RBC_TAG_REASONING_EFFORT, RBC_TAG_WORKERS
from rbc_reports import store
from time_utils import market_now

try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # dotenv not installed -- that's fine, falls through to os.environ

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

REPORT_TYPES = ("company", "sector", "macro", "roundup")
ACTIONS = ("initiation", "upgrade", "downgrade", "pt_raise", "pt_cut", "results_beat",
           "results_miss", "results_inline", "preview", "estimates", "event", "none")
TONES = ("bullish", "neutral", "bearish")
THEMES = (
    "banks", "insurance_asset_mgmt", "fintech_payments", "energy_oil_gas", "utilities_power",
    "clean_energy", "metals_mining", "gold_precious", "industrials", "aerospace_defense",
    "software", "tech_hardware", "semis_ai_datacenter", "internet_media", "telecom",
    "consumer_retail", "restaurants_leisure", "healthcare_biotech", "real_estate", "transport",
    "materials_chemicals", "macro_rates", "other",
)
# Field -> (allowed values, fallback for anything else)
_VOCAB = {
    "report_type": (REPORT_TYPES, "company"),
    "action": (ACTIONS, "none"),
    "tone": (TONES, "neutral"),
    "theme": (THEMES, "other"),
}
MAX_VIEW_CHARS = 400

_SYSTEM_PROMPT = (
    "You label RBC Capital Markets research reports from their title and teaser. "
    "Reply with ONLY a JSON object {\"reports\": [...]} holding one object per input "
    "report, in the same order, each with these keys: "
    "id (the input id, unchanged); "
    f"report_type (one of: {', '.join(REPORT_TYPES)} -- company = about one company; "
    "sector = an industry/peer group; macro = economy, rates, strategy, flows; "
    "roundup = a daily/weekly digest of many items); "
    f"action (one of: {', '.join(ACTIONS)} -- what this report does: rating initiation/"
    "upgrade/downgrade, price-target raise/cut, a results recap that beat/missed/was in "
    "line with expectations, a preview ahead of results, an estimate/model update, an "
    "event (deal, conference, data readout, management change), or none); "
    f"tone (one of: {', '.join(TONES)} -- RBC's stance on the subject's stock or "
    "sector: bullish if positive or the thesis is improving, bearish if negative or "
    "deteriorating, neutral if mixed or purely informational); "
    "rating (RBC's rating if the text states it, spelled out as \"Outperform\", "
    "\"Sector Perform\", \"Underperform\", \"Top Pick\" or \"Speculative Risk\" -- RBC "
    "abbreviates them OP/O, SP, UP/U -- else null; never guess); "
    f"theme (one of: {', '.join(THEMES)} -- the subject's industry). "
    "Judge only from the given text."
)


def _clean(label: dict) -> dict:
    out = {}
    for field, (allowed, fallback) in _VOCAB.items():
        value = str(label.get(field) or "").strip().lower()
        out[field] = value if value in allowed else fallback
    rating = label.get("rating")
    out["rating"] = str(rating).strip()[:40] if rating and str(rating).strip().lower() != "null" else None
    return out


def classify_batch(rows: list, client=None) -> dict:
    """rows: [(report_key, title, view, tickers_raw)]. Returns
    {report_key: cleaned labels} for every report the model answered; {} on
    any failure (the caller leaves those untagged)."""
    if client is None:
        if not OPENAI_API_KEY:
            return {}
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    lines = []
    for i, (_, title, view, tickers) in enumerate(rows):
        lines.append(json.dumps({"id": i, "title": title, "view": (view or "")[:MAX_VIEW_CHARS],
                                 "tickers": tickers or ""}, ensure_ascii=False))
    try:
        resp = client.chat.completions.create(
            model=RBC_TAG_MODEL,
            messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                      {"role": "user", "content": "\n".join(lines)}],
            response_format={"type": "json_object"},
            reasoning_effort=RBC_TAG_REASONING_EFFORT,
            timeout=120,
        )
        answers = json.loads(resp.choices[0].message.content).get("reports", [])
    except Exception:
        return {}
    out = {}
    for a in answers:
        if not isinstance(a, dict):
            continue
        try:
            idx = int(a.get("id"))
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(rows):
            out[rows[idx][0]] = _clean(a)
    return out


def tag_untagged(conn=None, limit: int | None = None, retag: bool = False,
                 client=None, dry_run: bool = False) -> dict:
    """Tags every report without a tag row (newest first; `retag` = every
    report). Returns {"configured", "pending", "tagged"}."""
    result = {"configured": bool(client or OPENAI_API_KEY), "pending": 0, "tagged": 0}
    if conn is None:
        if dry_run and not Path(store.DB_PATH).exists():
            return result
        conn = store.connect()
    rows = store.untagged_reports(conn, limit=limit, retag=retag)
    result["pending"] = len(rows)
    if not rows or not result["configured"] or dry_run:
        return result
    batches = [rows[i:i + RBC_TAG_BATCH_SIZE] for i in range(0, len(rows), RBC_TAG_BATCH_SIZE)]
    with ThreadPoolExecutor(max_workers=RBC_TAG_WORKERS) as pool:
        labelled = list(pool.map(lambda b: classify_batch(b, client=client), batches))
    now = market_now().isoformat()
    for labels in labelled:
        result["tagged"] += store.save_tags(conn, labels, RBC_TAG_MODEL, now, commit=False)
    conn.commit()
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Label stored RBC reports with an LLM.")
    p.add_argument("--limit", type=int, default=None, help="only the newest N untagged reports")
    p.add_argument("--retag", action="store_true", help="re-tag every report")
    p.add_argument("--dry-run", action="store_true", help="count what would be tagged, call nothing")
    args = p.parse_args(argv)
    r = tag_untagged(limit=args.limit, retag=args.retag, dry_run=args.dry_run)
    if not r["configured"]:
        print("OPENAI_API_KEY not set in .env -- nothing tagged")
        return 1
    print(f"{r['pending']} report(s) to tag" + ("" if args.dry_run else f", {r['tagged']} tagged"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
