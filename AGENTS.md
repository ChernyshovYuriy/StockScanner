# AGENTS.md

This repository is a scheduled personal trading system for Canadian equities.

Codex must optimize for:
1. minimal, reviewable diffs
2. operational safety
3. preserving current behavior unless the task explicitly requests behavior changes
4. respecting scope exactly

## Repository architecture

This repo intentionally uses three separate scheduled entrypoints.

- `main.py` runs at 4:30 PM and is the end-of-day pipeline service.
- `virtual_buy.py` runs at 9:45 AM and is the virtual entry execution service.
- `positions_monitor.py` runs at 3:50 PM and is the position monitoring / virtual exit service.

Do not merge these services unless the task explicitly requests that architecture change.

## Primary engineering rule

Make the minimum change required to satisfy the task.

Do not “improve” nearby code unless the prompt explicitly asks for it.

If you notice additional improvements, do not implement them. List them under a section called `Not implemented because out of scope`.

## Hard scope rules

Unless the task explicitly requests otherwise:

- preserve all existing comments verbatim
- do not remove comments for cleanup or style reasons
- do not rewrite comments for tone or wording
- do not convert constants into CLI arguments or config values
- do not rename files, functions, classes, variables, constants, or modules
- do not move code across files
- do not perform opportunistic refactors
- do not do formatting-only edits
- do not reorder imports unless required for correctness
- do not add dependencies unless explicitly requested
- do not replace existing working patterns just because another pattern looks cleaner
- do not update unrelated docs
- do not touch unrelated tests
- do not edit files outside the requested scope

If an extra change seems necessary, stop and explain why before making it.

## File change discipline

Before editing, inspect the repo and determine the smallest safe set of files to change.

When the task is non-trivial, first produce a file-by-file plan that includes:
- exact files to change
- exact reason each file must change
- exact behavior that will change
- explicit note of anything tempting but out of scope

Do not edit files that are not in the plan unless a genuine blocker appears. If that happens, explain the blocker and minimize expansion.

## Comment preservation policy

Comments are part of the maintained codebase and must be treated as intentional.

Rules:
- preserve existing comments exactly
- do not delete comments because they seem redundant
- do not shorten comments
- do not “clean up” comment wording
- do not replace comments with docstrings unless explicitly requested
- only change a comment when it becomes factually incorrect because of the requested code change
- when changing such a comment, keep the edit as small as possible

## Constants and configuration policy

This repo may intentionally use in-code constants.

Unless explicitly requested:
- do not convert constants to command-line arguments
- do not convert constants to environment variables
- do not move constants into config files
- do not add new config surface area for convenience
- do not widen configurability beyond the requested change

Exception:
- if the task explicitly asks for a new CLI flag, add only that specific flag and nothing broader

## Refactor policy

Refactoring is out of scope by default.

Unless explicitly requested:
- do not extract helper functions just for style
- do not introduce base classes or abstractions
- do not split files
- do not rename identifiers
- do not replace imperative code with a “cleaner” architecture
- do not modernize unrelated code
- do not remove duplication unless it directly blocks the requested task

If a tiny helper is absolutely required, keep it local and minimal.

## Behavior preservation policy

Preserve existing behavior unless the task explicitly requests behavior changes.

When a behavior change is requested:
- change only the requested behavior
- keep all unrelated paths working the same way
- prefer wrappers and small guardrails over deeper rewrites
- preserve outputs, filenames, and interfaces unless the task explicitly changes them

If you suspect existing behavior is buggy but fixing it is not requested:
- do not fix it silently
- mention it under `Not implemented because out of scope`

## Logging and operational changes

For operational hardening tasks:
- prefer minimal invasive changes
- keep output human-readable unless the task explicitly requires a format change
- preserve existing report generation unless asked to replace it
- do not redesign job orchestration when only logging or locking is requested
- keep scheduled entrypoints stable

## Documentation rules

When editing documentation:
- update only the sections needed for the requested change
- do not rewrite the entire README unless explicitly requested
- keep wording concrete and operational
- ensure docs match the actual current code, not aspirational future design
- include usage examples only for changed interfaces

## Testing and verification

Codex produces better results when it verifies work, so verification is expected when feasible. :contentReference[oaicite:0]{index=0}

After code changes:
- run the smallest relevant verification possible
- prefer targeted checks over broad unrelated runs
- if tests exist for touched code, run them
- if no tests exist, do basic validation such as syntax checks, import checks, or argument parsing sanity
- do not fix unrelated failing tests
- report what was run and what was not run

If verification cannot be run, say so clearly.

## Done criteria

A task is done only when all of the following are true:
- requested behavior is implemented
- no unrelated behavior was changed
- existing comments were preserved unless a factual correction was required
- the diff is minimal and reviewable
- touched file count is as small as reasonably possible
- documentation is updated if and only if the interface or behavior changed
- verification results are reported clearly
- extra ideas are listed separately and not implemented

## Required output format for Codex task summaries

At the end of a task, provide:
1. `Approach`
2. `Files changed`
3. `What changed`
4. `Verification`
5. `Not implemented because out of scope`
6. `Risks or follow-ups`

In `Not implemented because out of scope`, include any tempting extra improvements you intentionally did not make.

## Planning guidance

For harder or ambiguous tasks, plan first before editing. Codex supports reusable project guidance via `AGENTS.md`, and planning first is recommended for more complex work. :contentReference[oaicite:1]{index=1}

When asked for a plan-first workflow:
- do not start coding immediately
- inspect the repo
- produce a file-by-file plan
- wait for approval if the user asked for approval before edits

## Approval mindset

Assume narrow approval.

The existence of write access does not imply permission to make broad improvements.

Interpret every task literally and conservatively.

## Repo-specific boundaries

Unless explicitly requested otherwise, do not:
- replace file-based state with SQLite or another database
- change the three-service schedule
- merge `main.py`, `virtual_buy.py`, and `positions_monitor.py`
- redesign trading strategy logic
- alter buy/sell rules for “improvement”
- replace existing report outputs
- introduce frameworks or heavy abstractions

## Preferred implementation style

Prefer:
- small local edits
- explicit code
- stable names
- backward compatibility
- operational clarity
- incremental hardening

Avoid:
- broad cleanup
- style-driven rewrites
- architecture astronautics
- speculative future-proofing

## When in doubt

When in doubt:
- do less, not more
- preserve comments
- preserve constants
- preserve names
- preserve structure
- ask for clarification or stop with a concise explanation instead of making extra changes