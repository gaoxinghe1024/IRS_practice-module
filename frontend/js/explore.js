/* explore.js — 浏览全部页 */

async function loadExplore(q = "", genre = "") {
  const data = await api(`/api/anime/list?q=${encodeURIComponent(q)}&genre=${encodeURIComponent(genre)}`);
  $("#exploreList").innerHTML = data.items.length
    ? data.items.map(cardHTML).join("")
    : `<div class="empty">没有找到符合条件的作品</div>`;
  bindCards();
}

async function initGenres() {
  const data = await api("/api/anime/list?page_size=100");
  const set = new Set();
  data.items.forEach(a => a.genres.forEach(g => set.add(g)));
  $("#genreFilter").innerHTML = `<option value="">全部类型</option>` +
    [...set].sort().map(g => `<option>${g}</option>`).join("");
}

window.addEventListener("DOMContentLoaded", async () => {
  initPage("explore");
  if (!requireAuth()) return;
  if (!(await checkAuth())) { location.reload(); return; }
  $("#searchBtn").onclick = () => loadExplore($("#searchInput").value, $("#genreFilter").value);
  $("#searchInput").onkeydown = e => { if (e.key === "Enter") loadExplore($("#searchInput").value, $("#genreFilter").value); };
  $("#genreFilter").onchange = () => loadExplore($("#searchInput").value, $("#genreFilter").value);
  try {
    await initGenres();
    await loadExplore();
  } catch (e) { toast("后端未启动：请运行 uvicorn main:app"); }
});
