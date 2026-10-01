// Collection detail: items, taste profile, rule filters + algorithm 2 recommendations.
import { animeCard, animeUrl, api, coverImg, el, errorText, handleAuthError, main, metaLine, notInterestedButton,
  saveButton, showState, t, toast } from "../common.js";

const id = Number(main.dataset.collectionId);
const items = document.getElementById("item-list");
const profileBox = document.getElementById("profile");
const form = document.getElementById("filter-form");
const status = document.getElementById("rec-status");
const recs = document.getElementById("rec-list");
let relaxation = null;

async function loadCollection() {
  try {
    const c = await api(`/collections/${id}`);
    document.getElementById("collection-name").textContent = c.name;
    document.getElementById("collection-desc").textContent = c.description || "";
    document.title = `${c.name} · AniLogic`;
    items.replaceChildren();
    if (!c.items.length) showState(items, "empty", t("collection.empty"));
    for (const it of c.items) {
      items.append(el("li", {}, coverImg(it.cover_url, "", "thumb"),
        el("div", { class: "grow" }, el("a", { href: animeUrl(it.item_key) }, it.title), el("div", { class: "muted" }, metaLine(it))),
        el("button", { type: "button", class: "small secondary", onclick: async () => {
          try {
            await api(`/collections/${id}/items/${encodeURIComponent(it.item_key)}`, { method: "DELETE" });
            await loadCollection();
            await loadRecs();
          } catch (e) { toast(errorText(e), "error"); }
        } }, t("common.remove"))));
    }
  } catch (e) {
    if (!handleAuthError(e)) showState(items, "error", errorText(e), loadCollection);
  }
}

function renderProfile(p) {
  profileBox.replaceChildren();
  for (const ns of ["genre", "theme", "demographic", "studio"]) {
    const tags = p[ns] || [];
    if (!tags.length) continue;
    profileBox.append(el("h3", {}, ns), el("div", { class: "chips" },
      tags.map((x) => el("span", { class: "chip" }, `${x.tag} ${Math.round(x.share * 100)}%`))));
  }
  if (p.typical_episodes) profileBox.append(el("p", { class: "muted" }, t("collection.typical_length", { n: p.typical_episodes })));
  if (p.skipped_items?.length) profileBox.append(el("p", { class: "muted" }, t("collection.skipped", { n: p.skipped_items.length })));
}

function filters() {
  const body = { keywords: form.keywords.value.trim(), include_tags: [], min_score: null, max_episodes: null };
  if (form.genre.value) body.include_tags.push({ namespace: "genre", value: form.genre.value });
  if (form.min_score.value !== "") body.min_score = Number(form.min_score.value);
  if (form.max_episodes.value !== "") body.max_episodes = Number(form.max_episodes.value);
  return body;
}

function applyChange(change) {
  if ("max_episodes" in change) form.max_episodes.value = change.max_episodes ?? "";
  if ("min_score" in change) form.min_score.value = change.min_score ?? "";
  if ("keywords" in change) form.keywords.value = (change.keywords || []).join(", ");
  if ("include_tags" in change) form.genre.value = change.include_tags?.[0]?.[1] || "";
}

async function loadRecs() {
  showState(recs, "loading");
  status.replaceChildren();
  try {
    const res = await api(`/collections/${id}/recommendations`, { method: "POST",
      body: { ...filters(), relaxation_applied: relaxation && { rule_id: relaxation.rule_id, change: relaxation.change } } });
    renderProfile(res.profile);
    recs.replaceChildren();
    if (relaxation) status.append(el("p", { class: "notice warn" }, t("collection.relaxed_label", { reason: relaxation.reason })));
    for (const w of res.meta.warnings || []) if (res.status !== "no_match") status.append(el("p", { class: "muted" }, w));
    if (res.status === "empty_profile") { showState(recs, "empty", res.meta.warnings[0]); return; }
    if (res.status === "no_match") {
      const box = el("div", { class: "notice warn" }, el("p", {}, t("collection.no_match")));
      if (!res.relaxation_options.length) box.append(el("p", {}, t("collection.no_options")));
      for (const opt of res.relaxation_options) {
        box.append(el("p", {}, el("button", { type: "button", class: "small", onclick: () => {
          applyChange(opt.change);
          relaxation = { rule_id: opt.rule_id, change: opt.change, reason: opt.reason };
          loadRecs();
        } }, `${t("common.apply")}: ${opt.reason}`)));
      }
      status.append(box);
      return;
    }
    status.append(el("p", { class: "muted" }, `${t("collection.eligible", { n: res.meta.eligible_candidates })} · `
      + `${t("collection.scorer")}: ${res.meta.algorithm}`));
    for (const item of res.items) {
      recs.append(animeCard(item, { actions: [saveButton(item), notInterestedButton(item, { requestId: res.request_id,
        surface: "collection", algorithm: res.meta.algorithm, onDone: () => loadRecs() })] }));
    }
  } catch (e) {
    showState(recs, "error", errorText(e), loadRecs);
  }
}

async function initGenres() {
  try {
    const g = await api("/tags", { query: { namespace: "genre" } });
    for (const tag of g.items) form.genre.append(el("option", { value: tag.value }, tag.value));
  } catch (_) { /* optional */ }
}

form.addEventListener("submit", (ev) => { ev.preventDefault(); relaxation = null; loadRecs(); });
initGenres().then(async () => { await loadCollection(); await loadRecs(); });
