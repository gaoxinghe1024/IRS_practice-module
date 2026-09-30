// Onboarding: explicit liked tags + optional ratings of popular titles (cold start signals).
import { api, coverImg, el, errorText, handleAuthError, metaLine, showState, t } from "../common.js";

const chips = document.getElementById("tag-chips");
const titles = document.getElementById("title-list");
const errorBox = document.getElementById("welcome-error");
const selected = new Set();
const ratings = new Map();

async function load() {
  showState(chips, "loading");
  showState(titles, "loading");
  try {
    const [genres, themes, popular] = await Promise.all([
      api("/tags", { query: { namespace: "genre" } }), api("/tags", { query: { namespace: "theme" } }),
      api("/anime", { query: { sort: "popularity", page_size: 12 } })]);
    chips.replaceChildren();
    for (const [ns, list] of [["genre", genres.items], ["theme", themes.items.slice(0, 24)]]) {
      for (const tag of list) {
        const key = `${ns}|${tag.value}`;
        const b = el("button", { class: "chip", type: "button", "aria-pressed": "false" }, tag.value);
        b.addEventListener("click", () => {
          const on = !selected.has(key);
          on ? selected.add(key) : selected.delete(key);
          b.setAttribute("aria-pressed", String(on));
        });
        chips.append(b);
      }
    }
    titles.replaceChildren();
    for (const item of popular.items) {
      const select = el("select", { "aria-label": `${t("anime.your_rating")}: ${item.title}` },
        el("option", { value: "" }, "–"),
        ...Array.from({ length: 10 }, (_, i) => el("option", { value: String(10 - i) }, String(10 - i))));
      select.addEventListener("change", () => (select.value ? ratings.set(item.item_key, Number(select.value))
        : ratings.delete(item.item_key)));
      titles.append(el("li", {}, coverImg(item.cover_url, "", "thumb"),
        el("div", { class: "grow" }, el("strong", {}, item.title), el("div", { class: "muted" }, metaLine(item))), select));
    }
  } catch (e) {
    if (handleAuthError(e)) return;
    showState(chips, "error", errorText(e), load);
  }
}

document.getElementById("welcome-finish").addEventListener("click", async () => {
  errorBox.hidden = true;
  try {
    const liked = [...selected].map((k) => { const [namespace, value] = k.split("|"); return { namespace, value }; });
    await api("/me/preferences", { method: "PATCH", body: { liked_tags: liked } });
    for (const [key, rating] of ratings) {
      await api(`/anime/${encodeURIComponent(key)}/rating`, { method: "PUT", body: { rating } });
    }
    await api("/me/onboarding", { method: "POST", query: { state: "done" } });
    location.href = "/";
  } catch (e) {
    errorBox.textContent = errorText(e);
    errorBox.hidden = false;
  }
});
document.getElementById("welcome-skip").addEventListener("click", async () => {
  try { await api("/me/onboarding", { method: "POST", query: { state: "skipped" } }); } catch (_) { /* ignore */ }
  location.href = "/";
});
load();
