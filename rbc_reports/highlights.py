"""
rbc_reports/highlights.py
=========================
The day's highlights: an LLM reads one digest day's tagged company notes
together and picks the most bullish and most bearish of them (at most
RBC_HIGHLIGHTS_PER_SIDE each), each with a one-line reason taken from the
note's own text. The tagger's labels alone can't rank a day -- 41 notes
can all be "results_beat / bullish" -- so this is the one step that
compares notes against each other.

One call per digest day, after tagging (rbc_reports_service.py). A day is
redone only if more of its notes arrive later (a hand-saved digest). A
pick on the wrong side of the note's own tone (a "bullish" pick the
tagger read as bearish) is dropped, so the table never contradicts the
labels next to it. Like the tags, the picks are an LLM's reading and
UNVALIDATED -- the Track record table is what can score RBC's notes.
OPENAI_API_KEY comes from rbc_reports/tagger.py; unset -> nothing runs.

  python -m rbc_reports.highlights              # every digest day not yet done (last 7)
  python -m rbc_reports.highlights --days 30    # look further back
  python -m rbc_reports.highlights --redo       # redo those days (prompt change)
"""
from __future__ import annotations

import argparse
import json

from config import RBC_HIGHLIGHTS_PER_SIDE, RBC_TAG_MODEL, RBC_TAG_REASONING_EFFORT
from rbc_reports import store, tagger
from time_utils import market_now

MAX_VIEW_CHARS = 400
MAX_WHY_CHARS = 200
SIDES = ("bullish", "bearish")

_SYSTEM_PROMPT = (
    "You read one day of RBC Capital Markets company research notes (title and teaser, "
    "plus labels already assigned: action, tone, rating) and pick the ones an investor "
    "should look at first. Reply with ONLY a JSON object "
    "{\"bullish\": [...], \"bearish\": [...]}, each a list of at most "
    f"{RBC_HIGHLIGHTS_PER_SIDE} objects {{\"id\": <input id>, \"why\": <reason>}}, "
    "strongest first. Strongest means the clearest and biggest change in RBC's view: a "
    "rating change (upgrade/downgrade, an initiation at Outperform or Underperform), a "
    "large price-target or estimate change, a clear beat-and-raise or miss-and-cut, a "
    "thesis turning; prefer notes with concrete specifics over routine ones. A note goes "
    "in a list only if its stance is that direction; a list may be shorter, or empty. "
    "why: at most 20 words, what RBC said that makes it stand out, using only figures "
    "and facts in the given text -- never add any."
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS highlights (
    day TEXT NOT NULL,
    side TEXT NOT NULL,
    rank INTEGER NOT NULL,
    report_key TEXT NOT NULL,
    why TEXT,
    PRIMARY KEY (day, side, rank)
);
CREATE TABLE IF NOT EXISTS highlight_runs (
    day TEXT PRIMARY KEY,
    notes INTEGER NOT NULL,
    model TEXT,
    created_at TEXT NOT NULL
);
"""


def ensure_schema(conn) -> None:
    conn.executescript("BEGIN;" + _SCHEMA + "COMMIT;")


def day_notes(conn, day: str) -> list:
    """[(report_key, symbol, company, title, view, action, tone, rating)]:
    the tagged company notes from `day`'s digests (YYYY-MM-DD)."""
    return conn.execute(
        "SELECT r.report_key, (SELECT t.symbol FROM report_tickers t WHERE t.report_key = r.report_key "
        "ORDER BY t.position LIMIT 1), r.company, r.title, r.view, g.action, g.tone, g.rating "
        "FROM reports r JOIN report_tags g ON g.report_key = r.report_key "
        f"WHERE substr(r.digest_as_of, 1, 10) = ? AND r.{store.COMPANY_NOTE_SQL} "
        "AND g.report_type = 'company' ORDER BY r.published_at DESC", [day]).fetchall()


def pending_days(conn, days: int = 7, redo: bool = False) -> list:
    """The newest `days` digest days that need a run: none yet, or more
    tagged company notes now than when it ran. A day with untagged
    reports waits for the tagger."""
    out = []
    for (day,) in conn.execute(
            "SELECT DISTINCT substr(as_of, 1, 10) d FROM digests ORDER BY d DESC LIMIT ?", [days]):
        untagged = conn.execute(
            "SELECT COUNT(*) FROM reports r WHERE substr(r.digest_as_of, 1, 10) = ? AND NOT EXISTS "
            "(SELECT 1 FROM report_tags g WHERE g.report_key = r.report_key)", [day]).fetchone()[0]
        if untagged:
            continue
        n = len(day_notes(conn, day))
        done = conn.execute("SELECT notes FROM highlight_runs WHERE day = ?", [day]).fetchone()
        if n and (redo or done is None or n > done[0]):
            out.append(day)
    return out


def pick(notes: list, client=None) -> dict | None:
    """notes: day_notes() rows. Returns {"bullish": [(report_key, why)],
    "bearish": [...]}, or None on any failure (the day is retried next run)."""
    if client is None:
        if not tagger.OPENAI_API_KEY:
            return None
        from openai import OpenAI
        client = OpenAI(api_key=tagger.OPENAI_API_KEY)
    lines = [json.dumps({"id": i, "ticker": sym or "", "company": company or "", "title": title,
                         "view": (view or "")[:MAX_VIEW_CHARS], "action": action, "tone": tone,
                         "rating": rating}, ensure_ascii=False)
             for i, (_, sym, company, title, view, action, tone, rating) in enumerate(notes)]
    try:
        resp = client.chat.completions.create(
            model=RBC_TAG_MODEL,
            messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                      {"role": "user", "content": "\n".join(lines)}],
            response_format={"type": "json_object"},
            reasoning_effort=RBC_TAG_REASONING_EFFORT,
            timeout=300,
        )
        answer = json.loads(resp.choices[0].message.content)
    except Exception:
        return None
    if not isinstance(answer, dict):
        return None
    out = {}
    used = set()
    for side in SIDES:
        wrong_tone = "bearish" if side == "bullish" else "bullish"
        picks = []
        for p in answer.get(side) or []:
            if not isinstance(p, dict):
                continue
            try:
                idx = int(p.get("id"))
            except (TypeError, ValueError):
                continue
            if not 0 <= idx < len(notes) or idx in used or notes[idx][6] == wrong_tone:
                continue
            used.add(idx)
            picks.append((notes[idx][0], str(p.get("why") or "").strip()[:MAX_WHY_CHARS] or None))
            if len(picks) == RBC_HIGHLIGHTS_PER_SIDE:
                break
        out[side] = picks
    return out


def run(conn=None, days: int = 7, redo: bool = False, client=None, dry_run: bool = False) -> dict:
    """Picks highlights for every pending digest day. Returns
    {"configured", "pending", "done"} (counts of days)."""
    result = {"configured": bool(client or tagger.OPENAI_API_KEY), "pending": 0, "done": 0}
    conn = conn or store.connect()
    ensure_schema(conn)
    todo = pending_days(conn, days=days, redo=redo)
    result["pending"] = len(todo)
    if not result["configured"] or dry_run:
        return result
    for day in todo:
        notes = day_notes(conn, day)
        picks = pick(notes, client=client)
        if picks is None:
            continue
        conn.execute("DELETE FROM highlights WHERE day = ?", [day])
        conn.executemany(
            "INSERT INTO highlights(day, side, rank, report_key, why) VALUES(?,?,?,?,?)",
            [(day, side, i, key, why) for side in SIDES for i, (key, why) in enumerate(picks[side])])
        conn.execute("INSERT OR REPLACE INTO highlight_runs(day, notes, model, created_at) VALUES(?,?,?,?)",
                     [day, len(notes), RBC_TAG_MODEL, market_now().isoformat()])
        conn.commit()
        result["done"] += 1
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Pick each RBC digest day's most bullish/bearish notes.")
    p.add_argument("--days", type=int, default=7, help="how many of the newest digest days to consider")
    p.add_argument("--redo", action="store_true", help="redo days that already have highlights")
    p.add_argument("--dry-run", action="store_true", help="count the days, call nothing")
    args = p.parse_args(argv)
    r = run(days=args.days, redo=args.redo, dry_run=args.dry_run)
    if not r["configured"]:
        print("OPENAI_API_KEY not set in .env -- nothing picked")
        return 1
    print(f"{r['pending']} day(s) to do" + ("" if args.dry_run else f", {r['done']} done"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
