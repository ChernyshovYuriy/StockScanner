function makeSortable(table) {
  const tbody = table.querySelector("tbody");
  const headers = Array.from(table.querySelectorAll("thead th"));
  if (!tbody) return;

  const cellValue = (row, colIndex) => {
    const cell = row.children[colIndex];
    const text = cell ? cell.textContent.trim() : "";
    const num = parseFloat(text.replace(/[$%,+]/g, ""));
    return { text, num };
  };

  headers.forEach((th, colIndex) => {
    th.classList.add("sortable-col");
    th.addEventListener("click", () => {
      const ascending = th.dataset.sortDir !== "asc";
      const rows = Array.from(tbody.querySelectorAll("tr"));

      rows.sort((rowA, rowB) => {
        const a = cellValue(rowA, colIndex);
        const b = cellValue(rowB, colIndex);
        const bothNumeric = a.text !== "—" && b.text !== "—" && !isNaN(a.num) && !isNaN(b.num);
        const cmp = bothNumeric ? a.num - b.num : a.text.localeCompare(b.text);
        return ascending ? cmp : -cmp;
      });

      headers.forEach((h) => { h.dataset.sortDir = ""; h.classList.remove("sort-asc", "sort-desc"); });
      th.dataset.sortDir = ascending ? "asc" : "desc";
      th.classList.add(ascending ? "sort-asc" : "sort-desc");

      rows.forEach((row) => tbody.appendChild(row));
    });
  });
}

document.querySelectorAll("table.sortable").forEach(makeSortable);

document.addEventListener("click", (event) => {
  const btn = event.target.closest(".sell-btn");
  if (!btn) return;

  const ticker = btn.dataset.ticker;
  if (!confirm(`Sell ${ticker} now at market price?`)) return;

  btn.disabled = true;
  btn.textContent = "Selling…";

  fetch(`/api/positions/${encodeURIComponent(ticker)}/sell`, { method: "POST" })
    .then(async (res) => {
      const data = await res.json();
      if (res.ok && data.ok) {
        location.reload();
        return;
      }
      alert(data.message || "Sell failed.");
      btn.disabled = false;
      btn.textContent = "Sell";
    })
    .catch((err) => {
      alert("Request failed: " + err);
      btn.disabled = false;
      btn.textContent = "Sell";
    });
});
