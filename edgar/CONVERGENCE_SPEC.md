# Convergence Report — Design Spec

A `report` subcommand that, for a given ticker (or a whole watchlist), stacks
the independent "big money" signals and flags where several point the same way.
The premise: no single filing means much; *convergence* of independent parties
with money and information is the signal worth your attention. Output is a
research trigger, never a trade instruction.

Everything keys on CIK, so each signal is a lookup against the same identifier.

---

## The signals it stacks (in descending signal strength)

### 1. Activist / large stake — `SC 13D` (and `13D/A`)
- **Source:** daily-index scan, already routed in `scanner.py`.
- **What to extract from the filing body (NEW parsing needed):**
  - filer name (the "big money")
  - percent of class owned (Item on the cover page)
  - 13D vs 13G (13D = activist intent = stronger; 13G = passive)
- **Scoring:** fresh 13D in last ~90 days = STRONG. 13G = MEDIUM.
- **Why first:** timely (10-day filing), names a conviction buyer, often
  precedes change.

### 2. Clustered, opportunistic insider buys — `Form 4` code `P`
- **Source:** `insiders.py` (already isolates code-P buys).
- **Refinements NEEDED for signal quality:**
  - **Cluster:** >=2 distinct insiders buying within a rolling ~90-day window
    (you have `cluster_flag`; widen it to a window, not just "recent filings").
  - **Opportunistic filter:** down-weight buys flagged `aff10b5One=true` or
    footnoted as 10b5-1 (scheduled). Discretionary buys > scheduled.
  - **Merge split filings:** same owner+issuer+day across multiple Form 4s =
    one logical filing (the "first/second of two" case seen in MU).
  - **Role weight:** CEO/CFO buys > other officers > directors (optional).
- **Scoring:** cluster of opportunistic buys = STRONG. single opportunistic
  buy = MEDIUM. scheduled/10b5-1 buys = WEAK.

### 3. Institutional accumulation — `13F-HR`
- **Source:** new fetch (institutional manager holdings, quarterly).
- **What to extract:** for the target CIK, which managers report a position and
  whether it's NEW or INCREASED vs the prior quarter (requires storing two
  quarters and diffing — store.py is built for this).
- **Caveat baked into output:** up to 45 days stale. CONTEXT, not trigger.
- **Scoring:** multiple respected funds newly accumulating = MEDIUM (good
  corroboration, never timing). single fund = WEAK.

### 4. Fundamentals trend — `companyfacts`
- **Source:** `fundamentals.py` (already pulls latest; ADD prior-year quarter
  for YoY direction).
- **What to compute:** revenue & net-income YoY growth; margin; debt/equity.
  Reduce to a one-line read: "growing, profitable, lightly levered" vs not.
- **Role in convergence:** the QUALITY gate. Conviction buying into a
  *deteriorating* business is a different (often worse) story than buying into
  an improving one. Fundamentals don't add a "buy" vote; they qualify the
  others.

### 5. Recent catalyst — `8-K` (optional context line)
- **Source:** daily-index scan (already routed).
- **Role:** not scored; just surfaced. "Heads-up: 8-K filed 3 days ago" tells
  you news may explain a move. Parsing 8-K item codes is a later nicety.

---

## How it flags a convergence hit

Assign each signal a strength (STRONG=2, MEDIUM=1, WEAK=0) and require
**independence** — the point is multiple *different parties* agreeing:

```
parties_agreeing = count of DISTINCT signal categories scoring >= MEDIUM
                   among {13D, insider-buys, 13F-accumulation}
```

- `parties_agreeing >= 2`  AND  fundamentals not deteriorating  -> **CONVERGENCE FLAG**
- `parties_agreeing == 1`                                        -> "single signal — watch"
- otherwise                                                      -> no flag

Fundamentals are a GATE, not a vote: they can veto a flag (deteriorating
business) but don't create one. 8-K is shown but never scored.

Deliberately NOT doing: combining into a single 0-100 "score." A blended score
hides which parties agree and invites treating it as a price target. Show the
stack; let the human read it.

---

## Output shape (per ticker)

```
COEUR MINING (CDE)  CIK 215466
  13D/G        : SC 13D  filed 2026-05-12 by <Filer>, 7.8% stake   [STRONG]
  insider buys : 3 opportunistic buys, 2 insiders, last 60d        [STRONG]
  13F change   : 2 funds newly initiated last quarter               [MEDIUM]
  fundamentals : revenue +18% YoY, profitable, debt/equity 0.3      [GATE: ok]
  catalyst     : 8-K filed 2026-05-20 (not scored)
  -----------------------------------------------------------------
  >> CONVERGENCE FLAG: 3 independent parties positioning; fundamentals support
     (research trigger — not a recommendation)
```

When nothing converges, say so plainly ("no convergence; 1 weak signal") rather
than manufacturing significance.

---

## Build order (each step independently testable against live EDGAR)

1. **13D/G body parser** — extract filer + percent + 13D-vs-13G from the
   filing. Highest-value new capability; test against a real recent 13D.
2. **Insider refinements** — windowed clustering, 10b5-1 down-weighting, split-
   filing merge. Test fixtures already exist (the two MU documents).
3. **Fundamentals YoY** — add prior-year-quarter fetch + growth/margin/leverage
   one-liner.
4. **13F fetch + quarter-over-quarter diff** — store two quarters, diff for
   NEW/INCREASED. Most code; lowest signal; do last.
5. **`report` subcommand** — assemble the stack, apply the gate, print.

---

## Honest limits (keep these in the output and the README)

- Every signal is a LAGGED disclosure. This finds where conviction capital was
  *positioned*, not where it's going. You follow footprints.
- Convergence raises a prior and gets a name onto your desk earlier and more
  systematically than the crowd. It does not time entries or exits.
- The decision is yours: business quality, valuation, risk tolerance, and
  whether you understand the company. The filings supply attention, not answers.
- A single filing — buy or sell — is never enough to act on. Sells especially
  carry little signal (mostly 10b5-1 / diversification / liquidity).
