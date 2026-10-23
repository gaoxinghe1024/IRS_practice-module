// Developer tools (admin only): algorithm comparisons and run traces.
import { api, el, errorText, showState } from "../common.js";

document.getElementById("sim-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const out = document.getElementById("sim-out");
  showState(out, "loading");
  try {
    const res = await api(`/admin/compare/similar/${encodeURIComponent(document.getElementById("sim-key").value.trim())}`);
    out.replaceChildren(...Object.entries(res.scorers).map(([name, items]) => el("div", {}, el("h3", {}, name),
      el("ol", {}, items.map((i) => el("li", {}, `${i.title} (${i.score})`))))));
  } catch (e) { showState(out, "error", errorText(e)); }
});
document.getElementById("home-run").addEventListener("click", async () => {
  const out = document.getElementById("home-out");
  showState(out, "loading");
  try {
    const res = await api("/admin/compare/home");
    out.replaceChildren(...Object.entries(res.comparisons).map(([name, r]) => el("details", {},
      el("summary", {}, `${name} → ${r.algorithm}${r.fallback ? " (" + r.fallback + ")" : ""} · F=${r.objective.fitness}`),
      el("ol", {}, r.items.map((x) => el("li", {}, x))))));
  } catch (e) { showState(out, "error", errorText(e)); }
});
document.getElementById("trace-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const out = document.getElementById("trace-out");
  try {
    out.textContent = JSON.stringify(await api(`/admin/runs/${encodeURIComponent(document.getElementById("trace-id").value.trim())}`), null, 2);
  } catch (e) { out.textContent = errorText(e); }
});
