"""
press_release_tracker/llm_parser.py
=====================================
Turns one press-release RSS item's title + description into structured
fields (ticker, company, category, materiality, one-line summary) via the
OpenAI API. OPENAI_API_KEY is read from .env here (same self-contained
pattern demand_signals/darkpool.py uses for FINRA_CLIENT_ID/SECRET --
config.py itself doesn't load .env). Unconfigured -> parse_release()
returns None and the caller stores/emails the item with its raw RSS
fields only, same "degrade gracefully" convention as every other optional
credential in this repo.

Model: config.PRESS_RELEASE_LLM_MODEL, defaulting to OpenAI's cheapest/
smallest text tier -- this task (short-text classify + one-sentence
summary) doesn't need a frontier model. See config.py's own comment:
verify current pricing/model availability before relying on a cost
estimate; swapping tiers is a one-line change there.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # dotenv not installed -- that's fine, falls through to os.environ

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

try:
    from config import PRESS_RELEASE_LLM_MAX_DESCRIPTION_CHARS as MAX_DESC_CHARS
    from config import PRESS_RELEASE_LLM_MODEL as MODEL
except Exception:
    MODEL = "gpt-5-nano"
    MAX_DESC_CHARS = 2000

_SYSTEM_PROMPT = (
    "You extract structured data from Canadian company press-release wire "
    "items (GlobeNewswire and similar). Reply with ONLY a JSON object with "
    "these keys: "
    "ticker (string or null, e.g. \"OMI.V\" -- the TSX/TSXV ticker this "
    "release is about, only if it's actually identifiable from the text), "
    "company (string or null), "
    "category (one of: exploration_drilling, financing, ma_acquisition, "
    "earnings, contract_award, regulatory, personnel, other), "
    "materiality (one of: high, medium, low -- your judgment of how likely "
    "this news is to move the stock's price), "
    "summary (one plain-English sentence). "
    "If the ticker isn't stated in the text, use null rather than guessing."
)


def parse_release(title: str, description: str, categories: List[str]) -> Optional[dict]:
    """One OpenAI call -> {"ticker", "company", "category", "materiality",
    "summary"}, or None if OPENAI_API_KEY isn't set or the call fails
    (network error, malformed response, etc.) -- callers must treat that
    the same as "not parsed yet", never crash the run over one bad item.
    """
    if not OPENAI_API_KEY:
        return None

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    user_content = (
        f"Title: {title}\n"
        f"Categories: {', '.join(categories) if categories else '(none)'}\n"
        f"Description: {(description or '')[:MAX_DESC_CHARS]}"
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            timeout=30,
        )
        parsed = json.loads(resp.choices[0].message.content)
    except Exception:
        return None

    return {
        "ticker": parsed.get("ticker"),
        "company": parsed.get("company"),
        "category": parsed.get("category", "other"),
        "materiality": parsed.get("materiality", "low"),
        "summary": parsed.get("summary", ""),
    }
