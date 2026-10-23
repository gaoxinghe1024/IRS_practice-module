// Anime detail: facts, user actions, relations, similar titles (algorithm 1).
import { animeCard, animeUrl, api, coverImg, el, errorText, fmtEpisodes, loggedIn, logClick, main, myCollections,
  saveButton, showState, t, toast } from "../common.js";

const key = main.dataset.itemKey;
const root = document.getElementById("detail-root");
const simSection = document.getElementById("similar-section");
const simList = document.getElementById("similar-list");
let simK = 5;
let detail = null;

const unknown = () => el("span", { class: "unknown" }, t("common.unknown"));
const val = (v) => (v === null || v === undefined || v === "" || (Array.isArray(v) && !v.length) ? unknown()
  : Array.isArray(v) ? v.join(", ") : String(v));

function facts(d) {
  const rows = [
    ["anime.type", d.type], ["anime.episodes", d.episodes], ["anime.duration", d.duration_raw],
    ["anime.status", d.status], ["anime.aired", d.aired], ["anime.season", d.premiered],
    ["anime.studios", d.tags.studio], ["anime.source", d.source_material], ["anime.rating_cr", d.content_rating],
    ["anime.members", d.members !== null && d.members !== undefined ? d.members.toLocaleString() : null],
  ];
  return el("dl", { class: "facts" }, rows.flatMap(([k, v]) => [el("dt", {}, t(k)), el("dd", {}, val(v))]));
}

function ratingControl(d) {
  const current = d.user?.rating ?? null;
  const group = el("div", { class: "rating-buttons", role: "group", "aria-label": t("anime.your_rating") });
  for (let v = 1; v <= 10; v++) {
    group.append(el("button", { type: "button", "aria-pressed": String(current === v), onclick: () => setRating(v) },
      String(v)));
  }
  const clearBtn = el("button", { type: "button", class: "link-button", onclick: () => setRating(null) }, t("anime.clear_rating"));
  return el("div", { class: "field" }, el("label", {}, t("anime.your_rating")), group, current ? clearBtn : null);
}

async function setRating(v) {
  try {
    if (v === null) await api(`/anime/${encodeURIComponent(key)}/rating`, { method: "DELETE" });
    else await api(`/anime/${encodeURIComponent(key)}/rating`, { method: "PUT", body: { rating: v } });
    toast(v === null ? t("anime.clear_rating") : `${t("anime.your_rating")}: ${v}/10`);
    await load(false);
  } catch (e) { toast(errorText(e), "error"); }
}

function progressControl(d) {
  const p = d.user?.progress;
  const statuses = ["none", "plan_to_watch", "watching", "completed", "on_hold", "dropped"];
  const select = el("select", { id: "progress-status" },
    statuses.map((s) => el("option", { value: s, selected: (p?.status || "none") === s }, t(`anime.status.${s}`))));
  const eps = el("input", { id: "progress-eps", type: "number", min: "0", max: d.episodes ?? undefined,
    value: p?.watched_episodes ?? 0 });
  const save = el("button", { type: "button", class: "small", onclick: async () => {
    try {
      if (select.value === "none") await api(`/anime/${encodeURIComponent(key)}/progress`, { method: "DELETE" });
      else await api(`/anime/${encodeURIComponent(key)}/progress`, { method: "PUT",
        body: { status: select.value, watched_episodes: eps.value === "" ? null : Number(eps.value) } });
      toast(t("common.saved"));
      await load(false);
    } catch (e) { toast(errorText(e), "error"); }
  } }, t("common.save"));
  return el("div", { class: "field" }, el("label", { for: "progress-status" }, t("anime.progress")),
    el("div", { class: "inline" }, select,
      el("label", { class: "visually-hidden", for: "progress-eps" }, t("anime.watched_episodes")), eps,
      el("span", { class: "muted" }, "/ " + (d.episodes ?? "?")), save));
}

async function collectionsControl(d) {
  const cols = await myCollections(true);
  const member = new Set(d.user?.collections || []);
  const box = el("fieldset", {}, el("legend", {}, t("anime.collections")));
  for (const c of cols) {
    const cb = el("input", { type: "checkbox", checked: member.has(c.id) });
    cb.addEventListener("change", async () => {
      try {
        await api(`/collections/${c.id}/items/${encodeURIComponent(key)}`, { method: cb.checked ? "PUT" : "DELETE" });
        toast(cb.checked ? `${t("common.saved")}: ${c.name}` : `${t("common.remove")}: ${c.name}`);
      } catch (e) { cb.checked = !cb.checked; toast(errorText(e), "error"); }
    });
    box.append(el("label", { class: "checkbox" }, cb, c.name));
  }
  const name = el("input", { type: "text", maxlength: "60", "aria-label": t("anime.new_collection"),
    placeholder: t("anime.new_collection") });
  box.append(el("div", { class: "inline" }, name, el("button", { type: "button", class: "small secondary", onclick: async () => {
    if (!name.value.trim()) return;
    try {
      const c = await api("/collections", { method: "POST", body: { name: name.value.trim() } });
      await api(`/collections/${c.id}/items/${encodeURIComponent(key)}`, { method: "PUT" });
      toast(`${t("common.saved")}: ${c.name}`);
      await load(false);
    } catch (e) { toast(errorText(e), "error"); }
  } }, t("anime.add_new"))));
  return box;
}

function exclusionControl(d) {
  const exId = d.user?.excluded_exclusion_id;
  if (exId) {
    return el("p", { class: "notice warn" }, t("anime.not_interested_on"), " ",
      el("button", { type: "button", class: "small secondary", onclick: async () => {
        await api(`/exclusions/${exId}`, { method: "DELETE" });
        await load(false);
      } }, t("common.undo")));
  }
  return el("button", { type: "button", class: "small secondary", onclick: async () => {
    try {
      await api("/feedback", { method: "POST", body: { event_type: "not_interested", item_key: key, surface: "detail" } });
      await load(false);
    } catch (e) { toast(errorText(e), "error"); }
  } }, t("common.not_interested"));
}

function relations(d) {
  const sec = el("section", { class: "panel" }, el("h2", {}, t("anime.relations")));
  const wo = d.watch_order;
  if (wo?.watch_first?.length) {
    sec.append(el("p", {}, el("strong", {}, t("anime.watch_first") + ": "),
      ...wo.watch_first.flatMap((x, i) => [i ? ", " : "", el("a", { href: animeUrl(x.item_key) }, x.title || x.item_key)])));
  }
  if (wo?.prequels_released_later?.length) {
    sec.append(el("p", {}, el("strong", {}, t("anime.prequels_later") + ": "),
      ...wo.prequels_released_later.flatMap((x, i) => [i ? ", " : "",
        el("a", { href: animeUrl(x.item_key) }, x.title || x.item_key)])));
  }
  if (!d.relations.length) { sec.append(el("p", { class: "muted" }, t("anime.no_relations"))); return sec; }
  sec.append(el("ul", { class: "list" }, d.relations.map((r) => el("li", {},
    el("span", { class: "badge" }, r.relation_type.replaceAll("_", " ")),
    r.linkable ? el("a", { href: animeUrl(r.to_item) }, r.to_title) : el("span", {}, `${r.to_title || "?"} (${r.to_entity_type})`),
    el("small", { class: "muted" }, r.source_ref)))));
  return sec;
}

async function load(withSimilar = true) {
  try {
    const [d, rel] = await Promise.all([api(`/anime/${encodeURIComponent(key)}`),
      api(`/anime/${encodeURIComponent(key)}/relations`)]);
    detail = { ...d, watch_order: rel.watch_order };
    document.title = `${d.title} · AniLogic`;
    const synthetic = key.startsWith("synthetic:");
    const head = el("div", {},
      el("h1", {}, d.title),
      d.title_english && d.title_english !== d.title ? el("p", { class: "muted" }, d.title_english) : null,
      d.title_other ? el("p", { class: "muted" }, d.title_other) : null,
      el("p", {}, el("span", { class: "badge accent" }, d.catalog_score === null ? t("common.no_score") : `★ ${d.catalog_score}`),
        " ", el("small", { class: "muted" }, synthetic ? t("anime.score_note_synthetic") : t("anime.score_note"))),
      d.is_recommendable ? null : el("p", { class: "notice warn" }, t("anime.not_recommendable")),
      el("div", { class: "chips" }, [...(d.tags.genre || []), ...(d.tags.theme || []), ...(d.tags.demographic || [])]
        .map((g) => el("span", { class: "chip" }, g))));
    const side = el("div", { class: "stack" });
    if (loggedIn) {
      side.append(ratingControl(detail), progressControl(detail), await collectionsControl(detail), exclusionControl(detail));
    } else {
      side.append(el("p", { class: "notice" }, el("a", { href: "/login?next=" + encodeURIComponent(location.pathname) },
        t("anime.signin_to_rate"))));
    }
    const missing = d.missing_fields?.length ? el("p", { class: "muted" }, `${t("anime.missing")}: ${d.missing_fields.join(", ")}`) : null;
    root.replaceChildren(
      el("div", { class: "detail" }, coverImg(d.cover_url, d.title, "cover-lg"),
        el("div", {}, head, el("div", { class: "two-col" },
          el("div", {}, el("h2", {}, t("anime.synopsis")),
            el("p", {}, d.synopsis || t("anime.no_synopsis")),
            el("h2", {}, t("anime.details")), facts(d), missing),
          el("div", { class: "panel" }, side)))),
      relations(detail),
      d.characters.length ? el("section", { class: "panel" }, el("h2", {}, t("anime.characters")),
        el("div", { class: "chips" }, d.characters.map((c) => el("span", { class: "chip" }, `${c.name}${c.role ? " · " + c.role : ""}`)))) : null,
      el("details", { class: "panel evidence" }, el("summary", {}, t("anime.sources")),
        el("ul", {}, d.sources.map((s) => el("li", {}, `${s.kind} — ${s.source_type} — ${s.source_ref}`))),
        el("p", { class: "muted" }, `${t("common.dataset")}: ${d.dataset_mode} (${d.dataset_version})`)));
    if (withSimilar) loadSimilar();
  } catch (e) {
    showState(root, "error", errorText(e));
  }
}

async function loadSimilar() {
  simSection.classList.remove("hidden");
  showState(simList, "loading");
  const hide = document.getElementById("hide-watched");
  if (loggedIn) document.getElementById("hide-watched-wrap").classList.remove("hidden");
  try {
    const res = await api(`/anime/${encodeURIComponent(key)}/similar`,
      { query: { k: simK, hide_watched: loggedIn ? hide.checked : undefined } });
    simList.replaceChildren();
    const warn = document.getElementById("similar-warnings");
    warn.replaceChildren(...(res.meta.warnings || []).map((w) => el("p", { class: "notice warn" }, w)));
    if (!res.items.length) { showState(simList, "empty"); return; }
    for (const item of res.items) {
      const card = animeCard(item, { actions: [saveButton(item)] });
      card.querySelector("a.title").addEventListener("click", () => logClick(item, res.request_id, "similar"));
      simList.append(card);
    }
    document.getElementById("similar-more").textContent = simK === 5 ? t("common.more") : t("common.less");
  } catch (e) {
    showState(simList, "error", errorText(e), loadSimilar);
  }
}

document.getElementById("similar-more").addEventListener("click", () => { simK = simK === 5 ? 10 : 5; loadSimilar(); });
document.getElementById("hide-watched").addEventListener("change", loadSimilar);
load();
