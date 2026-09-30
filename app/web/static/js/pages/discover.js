// Discover: catalogue search / filters / pagination (URL-synced).
import { animeCard, api, el, errorText, readQuery, saveButton, showState, t, writeQuery } from "../common.js";

const form = document.getElementById("search-form");
const results = document.getElementById("results");
const count = document.getElementById("result-count");
const pager = document.getElementById("pager");
const FIELDS = ["q", "genres", "types", "max_episodes", "min_score", "sort"];
let page = Number(readQuery().page || 1);

async function initGenres() {
  try {
    const g = await api("/tags", { query: { namespace: "genre" } });
    for (const tag of g.items) form.genres.append(el("option", { value: tag.value }, `${tag.value} (${tag.count})`));
  } catch (_) { /* the select stays with "Any" */ }
  const q = readQuery();
  for (const f of FIELDS) if (q[f]) form[f].value = q[f];
}

async function search() {
  const params = Object.fromEntries(FIELDS.map((f) => [f, form[f].value.trim()]));
  writeQuery({ ...params, page: page > 1 ? page : "" });
  showState(results, "loading");
  pager.replaceChildren();
  try {
    const res = await api("/anime", { query: { ...params, page, page_size: 24 } });
    count.textContent = t("discover.results", { total: res.total });
    results.replaceChildren();
    if (!res.items.length) { showState(results, "empty", t("discover.no_results")); return; }
    for (const item of res.items) results.append(animeCard(item, { showReasons: false, actions: [saveButton(item)] }));
    const pages = Math.max(1, Math.ceil(res.total / res.page_size));
    pager.append(
      el("button", { class: "secondary small", type: "button", disabled: page <= 1, onclick: () => { page -= 1; search(); } },
        t("common.prev")),
      el("span", {}, t("common.page", { page, pages })),
      el("button", { class: "secondary small", type: "button", disabled: page >= pages,
        onclick: () => { page += 1; search(); } }, t("common.next")));
  } catch (e) {
    showState(results, "error", errorText(e), search);
  }
}

form.addEventListener("submit", (ev) => { ev.preventDefault(); page = 1; search(); });
document.getElementById("clear-filters").addEventListener("click", () => { form.reset(); page = 1; search(); });
initGenres().then(search);
