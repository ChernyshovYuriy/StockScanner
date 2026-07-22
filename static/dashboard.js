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
