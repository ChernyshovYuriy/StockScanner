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

function postSell(ticker, price) {
  const body = price === undefined ? {} : { price };
  return fetch(`/api/positions/${encodeURIComponent(ticker)}/sell`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(async (res) => ({ res, data: await res.json() }));
}

document.addEventListener("click", (event) => {
  const btn = event.target.closest(".sell-btn");
  if (!btn) return;

  const ticker = btn.dataset.ticker;
  if (!confirm(`Sell ${ticker} now at market price?`)) return;

  btn.disabled = true;
  btn.textContent = "Selling…";

  postSell(ticker)
    .then(({ res, data }) => {
      if (res.ok && data.ok) {
        location.reload();
        return;
      }

      // No live quote for this ticker (e.g. a stale/delisted symbol) —
      // offer a manual price instead of leaving the position stuck.
      if (data.error === "no_price") {
        const manual = prompt(
          `No live price available for ${ticker}. Enter a price to sell at manually, or Cancel:`
        );
        if (manual === null) {
          btn.disabled = false;
          btn.textContent = "Sell";
          return;
        }
        const price = parseFloat(manual);
        if (isNaN(price) || price <= 0) {
          alert("Enter a valid positive number.");
          btn.disabled = false;
          btn.textContent = "Sell";
          return;
        }
        postSell(ticker, price).then(({ res: res2, data: data2 }) => {
          if (res2.ok && data2.ok) {
            location.reload();
            return;
          }
          alert(data2.message || "Sell failed.");
          btn.disabled = false;
          btn.textContent = "Sell";
        });
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
