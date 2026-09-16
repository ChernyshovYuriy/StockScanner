"""
press_release_tracker/digest.py
=================================
Plain-text email bodies for the press-release tracker's two delivery
lanes (see press_release_service.py): "high" fires immediately -- one
email per poll that found a materiality='high' item, no batching, since
catching a market-moving catalyst fast is the whole point of this sleeve;
"batch" accumulates everything else (medium/low/unclassified) and
flushes at most once an hour (config.PRESS_RELEASE_BATCH_INTERVAL_MINUTES).
Both share the same per-item formatting and the same subject prefix, so
every email this sleeve sends can be filtered on one string.
"""
from __future__ import annotations

# Fixed, unique across every email this sleeve sends (immediate AND
# batched) so it can be filtered/searched on in an email client -- the
# user's own explicit ask, not just a cosmetic subject line.
SUBJECT_PREFIX = "Stock News Results"


def build_digest(rows: list[dict], kind: str = "batch") -> tuple[str, str]:
    """rows: dicts with guid/feed_url/title/link/pubdate/ticker/company/
    category/materiality/summary (the LLM fields are None when
    OPENAI_API_KEY isn't configured or the parse failed -- the raw
    title/link still go out either way).

    kind: "high" (the immediate lane) or "batch" (the hourly rollup) --
    controls only the subject wording; the body format is identical
    either way. Returns (subject, body).
    """
    n = len(rows)
    plural = "s" if n != 1 else ""
    if kind == "high":
        subject = f"{SUBJECT_PREFIX} — \U0001F6A8 HIGH: {n} item{plural}"
    else:
        subject = f"{SUBJECT_PREFIX} — hourly digest: {n} item{plural}"

    lines = []
    for r in rows:
        header = r.get("ticker") or r.get("company") or "(ticker unknown)"
        lines.append(f"[{header}] {r['title']}")
        if r.get("category"):
            lines.append(f"  category: {r['category']}  materiality: {r.get('materiality')}")
        if r.get("summary"):
            lines.append(f"  {r['summary']}")
        lines.append(f"  {r['link']}")
        lines.append(f"  published: {r['pubdate']}")
        lines.append("")

    body = "\n".join(lines) if lines else "No new items."
    return subject, body
