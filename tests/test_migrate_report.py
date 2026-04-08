"""
tests/test_migrate_report.py
============================
Tests for migrate_report.py — the one-time migration that rewrites historical
Position Monitor HTML reports to match the current design.

Verified behaviours:
  1. Old 'Sold: $X proceeds (±$Y gain/loss)' label  →  'Sold proceeds' + 'Realised P&L'
  2. Injects missing 'Total funds available' and 'Total gain/loss' lines
  3. Reverses chronological block order to reverse-chronological
  4. Plants <!-- MONITOR_SECTION_TOP --> anchor for future prepend runs
  5. Already-migrated (new-format) sections are left unchanged (idempotency)
  6. Blocks without a Funds State section (oldest format) pass through safely
  7. File-not-found returns 0 without raising
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# HTML builder helpers that replicate the *old* code's output exactly
# ---------------------------------------------------------------------------

_SPACER8 = (
    "<table width='100%' cellpadding='0' cellspacing='0' border='0'>"
    "<tr><td height='8' style='font-size:1px;line-height:1px'>&nbsp;</td></tr>"
    "</table>"
)
_DARK_HDR_START = (
    "<table width='100%' cellpadding='0' cellspacing='0' border='0'"
    " bgcolor='#1a1d2e'"
)
_DOC_CLOSE = (
    "\n</td></tr></table>\n</td></tr></table>\n</body>\n</html>"
)
_SECTION_ANCHOR = "<!-- MONITOR_SECTION_TOP -->"
_DAY_END = "<!-- MONITOR_DAY_END -->"
_PLACEHOLDER = "<!-- POSITION_MONITOR_PLACEHOLDER -->"


def _posmoni_header_html(date_str: str) -> str:
    """Minimal Position Monitor dark header banner (matches _header_banner output)."""
    return (
        f"{_DARK_HDR_START}"
        f" style='border-collapse:collapse;background:#1a1d2e;margin-bottom:16px'>"
        f"<tr><td bgcolor='#1a1d2e' style='padding:20px 24px;background:#1a1d2e'>"
        f"<p style='margin:0 0 4px 0;padding:0;font-family:Arial,Helvetica,sans-serif;"
        f"font-size:22px;font-weight:bold;color:#ffffff'>&#128202; Position Monitor</p>"
        f"<p style='margin:0;padding:0;font-family:Arial,Helvetica,sans-serif;"
        f"font-size:12px;color:#9099b8'>Daily Report &nbsp;&middot;&nbsp; {date_str}</p>"
        f"</td></tr></table>"
    )


def _old_funds_state_html(
    funds_before: float,
    funds_after: float,
    funds_gained: float,
    realized_pnl: float = 0.0,
) -> str:
    """
    Replicates the old _funds_summary_html() output (pre-2b9b4da).
    No 'Total funds available' or 'Total gain/loss' lines.
    """
    if funds_gained > 0:
        pnl_sign = "+" if realized_pnl >= 0 else ""
        pnl_color = "#0a7c4e" if realized_pnl >= 0 else "#c0152f"
        pnl_word = "gain" if realized_pnl >= 0 else "loss"
        action_html = (
            f"<li><b>Sold</b>: ${funds_gained:,.2f} proceeds &nbsp;"
            f"(<span style='color:{pnl_color};font-weight:bold'>"
            f"{pnl_sign}${realized_pnl:,.2f} {pnl_word}</span>)</li>"
            f"<li><b>Funds after sells</b>: ${funds_after:,.2f}</li>"
        )
    else:
        action_html = f"<li><b>Funds remaining</b>: ${funds_after:,.2f}</li>"

    return (
        "<div style='margin:16px 0;padding:14px 16px;border:1px solid #dde1ea;"
        "background:#ffffff;font-family:Arial,Helvetica,sans-serif;color:#1a1d2e'>"
        "<div style='font-size:13px;font-weight:bold;margin-bottom:8px;'>Funds State</div>"
        f"<ul style='margin:0;padding-left:18px;line-height:1.6'>"
        f"<li><b>Funds before monitor</b>: ${funds_before:,.2f}</li>"
        f"{action_html}"
        "</ul></div>"
    )


def _new_funds_state_html(
    funds_before: float,
    funds_after: float,
    funds_gained: float,
    realized_pnl: float = 0.0,
    unrealised_pnl: float = 0.0,
    unrealised_position_value: float = 0.0,
) -> str:
    """
    Replicates the current _funds_summary_html() output (post-2b9b4da).
    Includes 'Sold proceeds', 'Realised P&L', 'Total funds available',
    and 'Total gain/loss'.
    """
    def _fmt(pnl: float) -> str:
        sign = "+" if pnl >= 0 else "-"
        return f"{sign}${abs(pnl):,.2f}"

    if funds_gained > 0:
        rpnl_color = "#0a7c4e" if realized_pnl >= 0 else "#c0152f"
        sell_section = (
            f"<li><b>Sold proceeds</b>: ${funds_gained:,.2f}</li>"
            f"<li><b>Realised P&L</b>: "
            f"<span style='color:{rpnl_color};font-weight:bold'>"
            f"{_fmt(realized_pnl)}</span></li>"
            f"<li><b>Funds after sells</b>: ${funds_after:,.2f}</li>"
        )
    else:
        sell_section = f"<li><b>Funds remaining</b>: ${funds_after:,.2f}</li>"

    total_funds = funds_after + unrealised_position_value
    total_pnl = realized_pnl + unrealised_pnl
    tpnl_color = "#0a7c4e" if total_pnl >= 0 else "#c0152f"
    summary_section = (
        f"<li><b>Total funds available</b>: ${total_funds:,.2f}"
        f"<span style='font-size:10px;color:#888888'> (cash + open positions)</span></li>"
        f"<li><b>Total gain/loss</b>: "
        f"<span style='color:{tpnl_color};font-weight:bold'>"
        f"{_fmt(total_pnl)}</span></li>"
    )
    return (
        "<div style='margin:16px 0;padding:14px 16px;border:1px solid #dde1ea;"
        "background:#ffffff;font-family:Arial,Helvetica,sans-serif;color:#1a1d2e'>"
        "<div style='font-size:13px;font-weight:bold;margin-bottom:8px;'>Funds State</div>"
        f"<ul style='margin:0;padding-left:18px;line-height:1.6'>"
        f"<li><b>Funds before monitor</b>: ${funds_before:,.2f}</li>"
        f"{sell_section}"
        f"{summary_section}"
        "</ul></div>"
    )


def _make_day_block_intermediate(date_str: str, funds_html: str) -> str:
    """
    Build a day block as generated by the intermediate-format code
    (has MONITOR_DAY_END, no SECTION_ANCHOR in block itself).
    """
    return (
        _SPACER8 +
        _posmoni_header_html(date_str) +
        "<p>positions table placeholder</p>" +
        funds_html +
        _DAY_END
    )


def _make_day_block_old(date_str: str, funds_html: str = "") -> str:
    """
    Build a day block as generated by the oldest code
    (no MONITOR_DAY_END, no Funds State).
    """
    return (
        _SPACER8 +
        _posmoni_header_html(date_str) +
        "<p>positions table placeholder</p>" +
        funds_html
    )


def _make_intermediate_report(*day_blocks: str) -> str:
    """
    Build a multi-day report in the intermediate format (has SECTION_ANCHOR,
    blocks are already in reverse-chronological order, each ends with DAY_END).
    The newest day is listed first.
    """
    pipeline = "<!DOCTYPE html><html><head></head><body><p>Pipeline section</p>"
    pos_section = _SECTION_ANCHOR + "".join(day_blocks)
    return pipeline + pos_section + _DOC_CLOSE


def _make_old_format_report(*day_blocks_chron: str) -> str:
    """
    Build a multi-day report in the old format (no SECTION_ANCHOR).
    Blocks are in chronological order (oldest first) as the old append
    logic produced.  Block 1 carries embedded doc_close fragments.
    """
    pipeline = "<!DOCTYPE html><html><head></head><body><p>Pipeline section</p>"

    # The old code's first run: replace PLACEHOLDER with block+doc_close.
    # Subsequent runs: prepend to </body>.
    # Net result for 3 days:
    #   pipeline [block1]\n</td></tr></table>\n</td></tr></table>\n[block2][block3]</body></html>
    if not day_blocks_chron:
        return pipeline + _DOC_CLOSE

    first = day_blocks_chron[0]
    rest = day_blocks_chron[1:]

    # Mimic: content.replace(PLACEHOLDER, pos_block + _doc_close())
    # Then subsequent: content.replace("</body>", pos_block + "</body>")
    result = pipeline + first + "\n</td></tr></table>\n</td></tr></table>\n"
    for block in rest:
        result += block
    result += "</body></html>"
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMigrateReport:

    def _run(self, path: Path) -> int:
        from migrate_report import migrate_report as _migrate
        return _migrate(path)

    # ── 1. Sold label → Sold proceeds + Realised P&L ────────────────────────

    def test_profitable_sell_label_rewritten(self, tmp_path):
        """Old 'Sold: ... proceeds (gain)' → 'Sold proceeds' + 'Realised P&L'."""
        old_funds = _old_funds_state_html(1000.0, 1150.0, 1150.0, realized_pnl=150.0)
        block = _make_day_block_intermediate("20250101", old_funds)
        report = tmp_path / "report.html"
        report.write_text(_make_intermediate_report(block), encoding="utf-8")

        self._run(report)
        html = report.read_text(encoding="utf-8")

        assert "Sold proceeds" in html
        assert "Realised P&L" in html
        assert "+$150.00" in html
        # Old label must be gone
        assert "<b>Sold</b>:" not in html
        # " gain" / " loss" as inline P&L suffix must not appear
        assert " gain</span>" not in html

    def test_losing_sell_label_rewritten(self, tmp_path):
        """Old '$-X loss' inline format becomes separate '-$X' Realised P&L line."""
        old_funds = _old_funds_state_html(1000.0, 850.0, 850.0, realized_pnl=-150.0)
        block = _make_day_block_intermediate("20250102", old_funds)
        report = tmp_path / "report.html"
        report.write_text(_make_intermediate_report(block), encoding="utf-8")

        self._run(report)
        html = report.read_text(encoding="utf-8")

        assert "Sold proceeds" in html
        assert "Realised P&L" in html
        assert "-$150.00" in html
        assert "gained" not in html.lower()
        assert " loss" not in html

    def test_no_sell_block_unchanged(self, tmp_path):
        """No-sell blocks keep 'Funds remaining' and get Total lines injected."""
        old_funds = _old_funds_state_html(1000.0, 1000.0, 0.0, realized_pnl=0.0)
        block = _make_day_block_intermediate("20250103", old_funds)
        report = tmp_path / "report.html"
        report.write_text(_make_intermediate_report(block), encoding="utf-8")

        self._run(report)
        html = report.read_text(encoding="utf-8")

        assert "Funds remaining" in html
        assert "Sold" not in html

    # ── 2. Total funds / Total gain-loss injected when absent ────────────────

    def test_total_funds_available_injected(self, tmp_path):
        """'Total funds available' is added to old-format blocks."""
        old_funds = _old_funds_state_html(1000.0, 1200.0, 1200.0, realized_pnl=200.0)
        assert "Total funds available" not in old_funds  # confirm old format

        block = _make_day_block_intermediate("20250104", old_funds)
        report = tmp_path / "report.html"
        report.write_text(_make_intermediate_report(block), encoding="utf-8")

        self._run(report)
        html = report.read_text(encoding="utf-8")

        assert "Total funds available" in html
        # Cash approximation = funds_after = 1,200.00
        assert "1,200.00" in html

    def test_total_gain_loss_injected_for_sell(self, tmp_path):
        """'Total gain/loss' = realised P&L for old sell blocks (no unrealised data)."""
        old_funds = _old_funds_state_html(1000.0, 1300.0, 1300.0, realized_pnl=300.0)
        block = _make_day_block_intermediate("20250105", old_funds)
        report = tmp_path / "report.html"
        report.write_text(_make_intermediate_report(block), encoding="utf-8")

        self._run(report)
        html = report.read_text(encoding="utf-8")

        assert "Total gain/loss" in html
        assert "+$300.00" in html

    def test_total_gain_loss_zero_for_no_sell_block(self, tmp_path):
        """No-sell blocks get 'Total gain/loss: +$0.00' as historical approximation."""
        old_funds = _old_funds_state_html(1000.0, 1000.0, 0.0, realized_pnl=0.0)
        block = _make_day_block_intermediate("20250106", old_funds)
        report = tmp_path / "report.html"
        report.write_text(_make_intermediate_report(block), encoding="utf-8")

        self._run(report)
        html = report.read_text(encoding="utf-8")

        assert "Total gain/loss" in html
        assert "+$0.00" in html

    def test_historical_note_present(self, tmp_path):
        """Injected Total lines carry the 'historical, open positions excluded' note."""
        old_funds = _old_funds_state_html(1000.0, 1000.0, 0.0)
        block = _make_day_block_intermediate("20250107", old_funds)
        report = tmp_path / "report.html"
        report.write_text(_make_intermediate_report(block), encoding="utf-8")

        self._run(report)
        html = report.read_text(encoding="utf-8")

        assert "historical, open positions excluded" in html

    # ── 3. Reverse-chronological ordering ────────────────────────────────────

    def test_old_chron_format_reversed(self, tmp_path):
        """
        Reports written by the old appending code (oldest first) must have
        their day blocks reversed so the newest date comes first.
        """
        block1 = _make_day_block_old("20250101")
        block2 = _make_day_block_old("20250102")
        block3 = _make_day_block_old("20250103")

        report = tmp_path / "report.html"
        # Old format: chronological (oldest first)
        report.write_text(
            _make_old_format_report(block1, block2, block3),
            encoding="utf-8",
        )

        self._run(report)
        html = report.read_text(encoding="utf-8")

        pos1 = html.index("20250101")
        pos2 = html.index("20250102")
        pos3 = html.index("20250103")

        # After migration: newest (20250103) must come first
        assert pos3 < pos2 < pos1, (
            f"Expected newest-first but got: day3={pos3}, day2={pos2}, day1={pos1}"
        )

    def test_intermediate_format_order_preserved(self, tmp_path):
        """
        Reports already in reverse-chronological order must not be re-sorted.
        """
        block3 = _make_day_block_intermediate("20250103", "")
        block2 = _make_day_block_intermediate("20250102", "")
        block1 = _make_day_block_intermediate("20250101", "")

        # Intermediate format already has newest first
        report = tmp_path / "report.html"
        report.write_text(
            _make_intermediate_report(block3, block2, block1),
            encoding="utf-8",
        )

        self._run(report)
        html = report.read_text(encoding="utf-8")

        pos1 = html.index("20250101")
        pos2 = html.index("20250102")
        pos3 = html.index("20250103")

        assert pos3 < pos2 < pos1

    # ── 4. MONITOR_SECTION_TOP anchor ────────────────────────────────────────

    def test_section_anchor_planted_in_old_format(self, tmp_path):
        """Old-format reports get the MONITOR_SECTION_TOP anchor after migration."""
        block = _make_day_block_old("20250110")
        report = tmp_path / "report.html"
        report.write_text(_make_old_format_report(block), encoding="utf-8")

        content_before = report.read_text(encoding="utf-8")
        assert _SECTION_ANCHOR not in content_before

        self._run(report)
        html = report.read_text(encoding="utf-8")

        assert _SECTION_ANCHOR in html

    def test_section_anchor_not_duplicated_in_intermediate_format(self, tmp_path):
        """Intermediate-format reports keep exactly one SECTION_ANCHOR."""
        block = _make_day_block_intermediate("20250111", "")
        report = tmp_path / "report.html"
        report.write_text(_make_intermediate_report(block), encoding="utf-8")

        self._run(report)
        html = report.read_text(encoding="utf-8")

        assert html.count(_SECTION_ANCHOR) == 1

    # ── 5. Idempotency ───────────────────────────────────────────────────────

    def test_idempotent_on_already_migrated_report(self, tmp_path):
        """Running migration twice produces identical output."""
        new_funds = _new_funds_state_html(1000.0, 1200.0, 1200.0, realized_pnl=200.0)
        block = _make_day_block_intermediate("20250120", new_funds)
        report = tmp_path / "report.html"
        report.write_text(_make_intermediate_report(block), encoding="utf-8")

        self._run(report)
        after_first = report.read_text(encoding="utf-8")

        self._run(report)
        after_second = report.read_text(encoding="utf-8")

        assert after_first == after_second

    def test_new_format_labels_not_altered(self, tmp_path):
        """'Sold proceeds' and 'Realised P&L' labels are preserved unchanged."""
        new_funds = _new_funds_state_html(1000.0, 1150.0, 1150.0, realized_pnl=150.0)
        block = _make_day_block_intermediate("20250121", new_funds)
        report = tmp_path / "report.html"
        report.write_text(_make_intermediate_report(block), encoding="utf-8")

        self._run(report)
        html = report.read_text(encoding="utf-8")

        assert "Sold proceeds" in html
        assert "Realised P&L" in html
        assert "<b>Sold</b>:" not in html
        # 'Total funds available' from the new-format generator uses the live note
        assert "cash + open positions" in html

    # ── 6. Blocks without Funds State (oldest format) ────────────────────────

    def test_block_without_funds_state_passes_through(self, tmp_path):
        """Day blocks that have no Funds State card are not corrupted."""
        block = _make_day_block_old("20250130")  # no funds HTML
        report = tmp_path / "report.html"
        report.write_text(_make_old_format_report(block), encoding="utf-8")

        self._run(report)
        html = report.read_text(encoding="utf-8")

        assert "20250130" in html
        assert "positions table placeholder" in html

    # ── 7. Edge cases ────────────────────────────────────────────────────────

    def test_returns_zero_for_missing_file(self, tmp_path):
        """migrate_report returns 0 and does not raise for a non-existent path."""
        from migrate_report import migrate_report as _migrate
        result = _migrate(tmp_path / "nonexistent.html")
        assert result == 0

    def test_returns_zero_for_file_with_no_position_monitor(self, tmp_path):
        """Files that contain only a pipeline report (no pos monitor) return 0."""
        from migrate_report import migrate_report as _migrate
        report = tmp_path / "report.html"
        report.write_text(
            "<!DOCTYPE html><html><body><p>Pipeline only</p>"
            f"{_PLACEHOLDER}</body></html>",
            encoding="utf-8",
        )
        result = _migrate(report)
        assert result == 0

    def test_returns_block_count(self, tmp_path):
        """migrate_report returns the number of day blocks processed."""
        blocks = [
            _make_day_block_intermediate(f"202501{i:02d}", "")
            for i in range(1, 4)
        ]
        report = tmp_path / "report.html"
        report.write_text(_make_intermediate_report(*blocks), encoding="utf-8")

        result = self._run(report)
        assert result == 3

    def test_day_end_markers_present_after_migration(self, tmp_path):
        """Every day block ends with <!-- MONITOR_DAY_END --> after migration."""
        block1 = _make_day_block_old("20250201")  # no DAY_END
        block2 = _make_day_block_old("20250202")
        report = tmp_path / "report.html"
        report.write_text(_make_old_format_report(block1, block2), encoding="utf-8")

        self._run(report)
        html = report.read_text(encoding="utf-8")

        # Each of the 2 day blocks should now end with the marker
        assert html.count(_DAY_END) == 2

    def test_multiple_blocks_all_get_total_lines(self, tmp_path):
        """All day blocks in a multi-day report receive Total lines."""
        blocks = [
            _make_day_block_intermediate(
                f"202502{i:02d}",
                _old_funds_state_html(1000.0, 1000.0 + i * 50, i * 50, realized_pnl=i * 10),
            )
            for i in range(1, 4)
        ]
        report = tmp_path / "report.html"
        report.write_text(_make_intermediate_report(*blocks), encoding="utf-8")

        self._run(report)
        html = report.read_text(encoding="utf-8")

        assert html.count("Total funds available") == 3
        assert html.count("Total gain/loss") == 3
