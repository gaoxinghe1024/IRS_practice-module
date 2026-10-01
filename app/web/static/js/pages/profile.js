// Profile: explicit preferences, rating history, watch progress, exclusions (undo).
import { animeUrl, api, coverImg, el, errorText, handleAuthError, showState, t, toast } from "../common.js";

const form = document.getElementById("prefs-form");
const chipsBox = document.getElementById("liked-chips");
const liked = new Set();

async function load() {
  try {
    const [me, genres, themes] = await Promise.all([api("/me"), api("/tags", { query: { namespace: "genre" } }),
      api("/tags", { query: { namespace: "theme" } })]);
    document.getElementById("account-line").textContent =
      `${me.user.username} · ${me.stats.ratings} ${t("profile.ratings").toLowerCase()} · ${me.stats.collections} ${t("nav.collections").toLowerCase()}`;
    for (const x of me.preferences.liked_tags) liked.add(`${x.namespace}|${x.value}`);
    chipsBox.replaceChildren();
    for (const [ns, list] of [["genre", genres.items], ["theme", themes.items.slice(0, 30)]]) {
      for (const tag of list) {
        const key = `${ns}|${tag.value}`;
        const b = el("button", { class: "chip", type: "button", "aria-pressed": String(liked.has(key)) }, tag.value);
        b.addEventListener("click", () => {
          liked.has(key) ? liked.delete(key) : liked.add(key);
          b.setAttribute("aria-pressed", String(liked.has(key)));
        });
        chipsBox.append(b);
      }
    }
    form.length.value = me.preferences.length_preference;
    form.hide_watched.checked = me.preferences.hide_watched_in_similar;
  } catch (e) { if (!handleAuthError(e)) showState(chipsBox, "error", errorText(e)); }
  loadRatings();
  loadProgress();
  loadExclusions();
}

async function loadRatings() {
  const box = document.getElementById("ratings");
  showState(box, "loading");
  try {
    const res = await api("/me/ratings", { query: { limit: 100 } });
    box.replaceChildren();
    if (!res.items.length) { showState(box, "empty", t("profile.no_ratings")); return; }
    for (const r of res.items) {
      box.append(el("li", {}, coverImg(r.cover_url, "", "thumb"), el("a", { class: "grow", href: animeUrl(r.item_key) }, r.title),
        el("span", { class: "badge accent" }, `${r.rating}/10`),
        el("button", { type: "button", class: "small secondary", onclick: async () => {
          await api(`/anime/${encodeURIComponent(r.item_key)}/rating`, { method: "DELETE" });
          loadRatings();
        } }, t("common.remove"))));
    }
  } catch (e) { showState(box, "error", errorText(e), loadRatings); }
}

async function loadProgress() {
  const box = document.getElementById("progress");
  try {
    const res = await api("/me/progress");
    box.replaceChildren();
    if (!res.items.length) { showState(box, "empty", t("profile.no_progress")); return; }
    for (const p of res.items) {
      box.append(el("li", {}, el("a", { class: "grow", href: animeUrl(p.item_key) }, p.title),
        el("span", { class: "badge" }, t(`anime.status.${p.status}`)),
        el("span", { class: "muted" }, `${p.watched_episodes}/${p.episodes ?? "?"}`)));
    }
  } catch (e) { showState(box, "error", errorText(e), loadProgress); }
}

async function loadExclusions() {
  const box = document.getElementById("exclusions");
  try {
    const res = await api("/me/exclusions");
    box.replaceChildren();
    if (!res.items.length) { showState(box, "empty", t("profile.no_exclusions")); return; }
    for (const x of res.items) {
      const label = x.kind === "item" ? el("a", { href: animeUrl(x.item_key) }, x.title) : el("span", {}, `${x.tag.namespace}: ${x.tag.value}`);
      box.append(el("li", {}, el("span", { class: "grow" }, label), el("button", { type: "button", class: "small secondary",
        onclick: async () => { await api(`/exclusions/${x.id}`, { method: "DELETE" }); loadExclusions(); } }, t("common.undo"))));
    }
  } catch (e) { showState(box, "error", errorText(e), loadExclusions); }
}

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  try {
    const liked_tags = [...liked].map((k) => { const [namespace, value] = k.split("|"); return { namespace, value }; });
    await api("/me/preferences", { method: "PATCH", body: { liked_tags, length_preference: form.length.value,
      hide_watched_in_similar: form.hide_watched.checked } });
    toast(t("common.saved"));
  } catch (e) { toast(errorText(e), "error"); }
});
load();
