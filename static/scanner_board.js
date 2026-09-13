/*
 * static/scanner_board.js
 * =========================
 * "Screen & sort" panel for the Scanner Board (/scanner) — client-side
 * multi-criteria filter and multi-column sort over rows already sent to
 * the page. No fetch, no new server route, no computed score: every
 * criterion reads one raw column the table already shows (see
 * scanner_dashboard_data.scanner_criteria_columns(), the single source
 * of truth for which columns this panel offers and what kind each one
 * is), and every sort just reorders the same rows. Page-scoped — no
 * other dashboard page loads this file.
 *
 * Data flow: the template embeds the full row set and the column
 * metadata as two <script type="application/json"> blocks (avoids
 * re-parsing formatted/badge text back out of the DOM); this file reads
 * those once, builds the panel UI from the column metadata, and on
 * Apply reorders/hides the existing <tr data-ticker="..."> elements to
 * match — the table markup itself (badges, glyphs, formatting) is never
 * regenerated, only reordered and shown/hidden.
 */
(function () {
  const rowsDataEl = document.getElementById("scanner-rows-data");
  const columnsDataEl = document.getElementById("scanner-columns-data");
  const tbody = document.querySelector(".scanner-table tbody");
  if (!rowsDataEl || !columnsDataEl || !tbody) return; // no data (e.g. empty board) -- nothing to wire up

  const ROWS = JSON.parse(rowsDataEl.textContent);
  const COLUMNS = JSON.parse(columnsDataEl.textContent);
  const COLUMNS_BY_KEY = Object.fromEntries(COLUMNS.map((c) => [c.key, c]));

  const domRowByTicker = new Map();
  tbody.querySelectorAll("tr[data-ticker]").forEach((tr) => domRowByTicker.set(tr.dataset.ticker, tr));

  const NUMBER_OPS = [
    { value: "gte", label: "≥" }, { value: "lte", label: "≤" },
    { value: "gt", label: ">" }, { value: "lt", label: "<" }, { value: "eq", label: "=" },
  ];
  const LABEL_OPS = [{ value: "eq", label: "is" }, { value: "neq", label: "is not" }];
  const TEXT_OPS = [{ value: "contains", label: "contains" }];

  function opsFor(col) {
    if (col.kind === "number") return NUMBER_OPS;
    if (col.kind === "label") return LABEL_OPS;
    return TEXT_OPS;
  }

  function fillSelect(select, options) {
    select.replaceChildren();
    options.forEach((o) => {
      const opt = document.createElement("option");
      opt.value = o.value;
      opt.textContent = o.label;
      select.appendChild(opt);
    });
  }

  function buildColumnSelect() {
    const select = document.createElement("select");
    const groups = new Map();
    COLUMNS.forEach((c) => {
      if (!groups.has(c.group)) {
        const optgroup = document.createElement("optgroup");
        optgroup.label = c.group;
        groups.set(c.group, optgroup);
        select.appendChild(optgroup);
      }
      const opt = document.createElement("option");
      opt.value = c.key;
      opt.textContent = c.label;
      groups.get(c.group).appendChild(opt);
    });
    return select;
  }

  // ---- Criteria rows ("column op value") ----
  const criteriaList = document.getElementById("criteria-list");
  const sortList = document.getElementById("sort-list");
  const MAX_CRITERIA = 6;
  const MAX_SORT = 3;

  function addCriterionRow() {
    if (criteriaList.children.length >= MAX_CRITERIA) return;
    const row = document.createElement("div");
    row.className = "criteria-row";

    const colSelect = buildColumnSelect();
    colSelect.className = "criteria-col";
    const opSelect = document.createElement("select");
    opSelect.className = "criteria-op";
    const valueSlot = document.createElement("span");

    function rebuildForColumn() {
      const col = COLUMNS_BY_KEY[colSelect.value];
      fillSelect(opSelect, opsFor(col));
      valueSlot.replaceChildren();
      let input;
      if (col.kind === "label") {
        input = document.createElement("select");
        fillSelect(input, col.options.map((v) => ({ value: v, label: v })));
      } else {
        input = document.createElement("input");
        input.type = col.kind === "number" ? "number" : "text";
        if (col.kind === "number") input.step = "any";
        input.placeholder = col.kind === "number" ? "value" : "text";
      }
      input.className = "criteria-value";
      valueSlot.appendChild(input);
    }

    colSelect.addEventListener("change", rebuildForColumn);
    rebuildForColumn();

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "criteria-remove";
    removeBtn.textContent = "✕";
    removeBtn.title = "Remove this criterion";
    removeBtn.addEventListener("click", () => row.remove());

    row.append(colSelect, opSelect, valueSlot, removeBtn);
    criteriaList.appendChild(row);
  }

  function addSortRow() {
    if (sortList.children.length >= MAX_SORT) return;
    const row = document.createElement("div");
    row.className = "criteria-row";

    const colSelect = buildColumnSelect();
    colSelect.className = "sort-col";
    const dirSelect = document.createElement("select");
    dirSelect.className = "sort-dir";
    fillSelect(dirSelect, [
      { value: "asc", label: "Ascending" },
      { value: "desc", label: "Descending" },
    ]);

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "criteria-remove";
    removeBtn.textContent = "✕";
    removeBtn.title = "Remove this sort column";
    removeBtn.addEventListener("click", () => row.remove());

    row.append(colSelect, dirSelect, removeBtn);
    sortList.appendChild(row);
  }

  // ---- Matching / ordering ----
  function matchesCriterion(row, colKey, op, rawValue) {
    const col = COLUMNS_BY_KEY[colKey];
    const cellValue = row[colKey];

    if (col.kind === "number") {
      if (rawValue === "" || rawValue === null) return true; // no value typed yet -- don't exclude anything
      if (cellValue === null || cellValue === undefined) return false; // no reading for this ticker -- can't match a numeric test
      const num = parseFloat(rawValue);
      if (isNaN(num)) return true;
      switch (op) {
        case "gte": return cellValue >= num;
        case "lte": return cellValue <= num;
        case "gt": return cellValue > num;
        case "lt": return cellValue < num;
        case "eq": return cellValue === num;
        default: return true;
      }
    }

    if (col.kind === "label") {
      return op === "neq" ? cellValue !== rawValue : cellValue === rawValue;
    }

    // text (ticker): substring, case-insensitive
    if (!rawValue) return true;
    return (cellValue || "").toLowerCase().includes(rawValue.toLowerCase());
  }

  function readCriteria() {
    return Array.from(criteriaList.children).map((row) => ({
      col: row.querySelector(".criteria-col").value,
      op: row.querySelector(".criteria-op").value,
      val: row.querySelector(".criteria-value").value,
    }));
  }

  function readSorts() {
    return Array.from(sortList.children).map((row) => ({
      col: row.querySelector(".sort-col").value,
      dir: row.querySelector(".sort-dir").value,
    }));
  }

  // Sort key for a label column is its position in that column's own
  // Enum declaration order (scanner_criteria_columns()'s `options`) --
  // NOT a designed bullish-to-bearish rank. Several of the Board's Enums
  // happen to declare their bullish member first (MACDCross, DIBias,
  // TrendHealth...), but not all of them (e.g. ForceZone declares
  // Negative before Positive) -- so "Ascending" on a label column means
  // "in the order this column's own values are defined," not "most
  // bullish first." A missing/unrecognized value sorts last.
  function sortKey(row, colKey) {
    const col = COLUMNS_BY_KEY[colKey];
    const v = row[colKey];
    if (col.kind === "number") return v === null || v === undefined ? -Infinity : v;
    if (col.kind === "label") {
      const idx = col.options.indexOf(v);
      return idx === -1 ? col.options.length : idx;
    }
    return (v || "").toLowerCase();
  }

  const countEl = document.getElementById("criteria-count");

  function setCount(n) {
    if (countEl) countEl.textContent = `Showing ${n} of ${ROWS.length} tickers`;
  }

  function applyCriteria() {
    const criteria = readCriteria();
    const sorts = readSorts();

    const matched = ROWS.filter((row) => criteria.every((c) => matchesCriterion(row, c.col, c.op, c.val)));

    if (sorts.length) {
      matched.sort((a, b) => {
        for (const s of sorts) {
          const ka = sortKey(a, s.col);
          const kb = sortKey(b, s.col);
          const cmp = typeof ka === "string" ? ka.localeCompare(kb) : ka - kb;
          if (cmp !== 0) return s.dir === "desc" ? -cmp : cmp;
        }
        return 0;
      });
    }

    const matchedTickers = new Set(matched.map((r) => r.ticker));
    matched.forEach((r) => {
      const tr = domRowByTicker.get(r.ticker);
      if (tr) tbody.appendChild(tr);
    });
    domRowByTicker.forEach((tr, ticker) => {
      tr.hidden = !matchedTickers.has(ticker);
    });

    setCount(matched.length);
  }

  function resetCriteria() {
    criteriaList.replaceChildren();
    sortList.replaceChildren();
    addCriterionRow();
    addSortRow();

    // Restore the server's own order (ORDER BY ticker) and show everything.
    ROWS.forEach((r) => {
      const tr = domRowByTicker.get(r.ticker);
      if (tr) {
        tr.hidden = false;
        tbody.appendChild(tr);
      }
    });
    setCount(ROWS.length);
  }

  document.getElementById("add-criterion-btn").addEventListener("click", addCriterionRow);
  document.getElementById("add-sort-btn").addEventListener("click", addSortRow);
  document.getElementById("apply-criteria-btn").addEventListener("click", applyCriteria);
  document.getElementById("reset-criteria-btn").addEventListener("click", resetCriteria);

  // Enter inside any criteria/sort input or select applies immediately,
  // without needing a <form>/submit (the buttons above are all type="button"
  // so they never interact with form submission semantics).
  document.querySelector(".criteria-panel").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      applyCriteria();
    }
  });

  addCriterionRow();
  addSortRow();
  setCount(ROWS.length);
})();
