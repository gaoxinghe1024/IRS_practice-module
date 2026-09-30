// Shared frontend module (repo rule: cross-page logic lives in common.js, one module per page).
// Everything user- or data-provided is inserted with textContent, never as HTML.

const I18N = JSON.parse(document.getElementById("i18n")?.textContent || "{}");

export function t(key, params = {}) {
  let s = I18N[key] ?? key;
  for (const [k, v] of Object.entries(params)) s = s.replaceAll(`{${k}}`, String(v));
  return s;
}

export const main = document.getElementById("main");
export const loggedIn = document.body.dataset.loggedIn === "true";
export const datasetMode = document.body.dataset.datasetMode;

// ------------------------------------------------------------------ API client
export class ApiError extends Error {
  constructor(status, code, message, details = {}, requestId = null) {
    super(message);
    this.status = status;
    this.code = code;
    this.details = details;
    this.requestId = requestId;
  }
}

function cookie(name) {
  const m = document.cookie.match(new RegExp("(?:^|; )" + name + "=([^;]*)"));
  return m ? decodeURIComponent(m[1]) : "";
}

export async function api(path, { method = "GET", body, query, raw = false, form } = {}) {
  let url = "/api/v1" + path;
  if (query) {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(query)) {
      if (v === undefined || v === null || v === "" || (Array.isArray(v) && !v.length)) continue;
      q.set(k, Array.isArray(v) ? v.join(",") : v);
    }
    const s = q.toString();
    if (s) url += "?" + s;
  }
  const headers = { Accept: "application/json" };
  if (method !== "GET") headers["X-CSRF-Token"] = cookie("anilogic_csrf");
  let payload;
  if (form) payload = form;
  else if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(url, { method, headers, body: payload, credentials: "same-origin" });
  } catch (e) {
    throw new ApiError(0, "network", t("error.network"));
  }
  if (res.status === 204) return null;
  if (!res.ok) {
    let err = {};
    try { err = (await res.json()).error || {}; } catch (_) { /* non-JSON error */ }
    throw new ApiError(res.status, err.code || "http_error", err.message || t("common.error"), err.details || {},
      err.request_id || res.headers.get("x-request-id"));
  }
  if (raw) return res;
  return res.json();
}

export function errorText(e) {
  if (e instanceof ApiError) {
    const key = "error." + e.code;
    const base = I18N[key] ? t(key) : e.message;
    return e.requestId ? `${base} (ref ${e.requestId})` : base;
  }
  return t("common.error");
}

export function handleAuthError(e) {
  if (e instanceof ApiError && e.status === 401) {
    location.href = "/login?next=" + encodeURIComponent(location.pathname + location.search);
    return true;
  }
  return false;
}

// ------------------------------------------------------------------ DOM helpers
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2).toLowerCase(), v);
    else if (v === true) node.setAttribute(k, "");
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

export function showState(container, kind, message, retry) {
  clear(container);
  const box = el("div", { class: `state ${kind}`, role: kind === "error" ? "alert" : "status" },
    message || t(kind === "loading" ? "common.loading" : kind === "error" ? "common.error" : "common.empty"));
  if (retry) box.append(" ", el("button", { class: "small secondary", type: "button", onclick: retry }, t("common.retry")));
  container.append(box);
  return box;
}

export function toast(message, kind = "info", action) {
  const region = document.getElementById("toast-region");
  const node = el("div", { class: `toast ${kind === "error" ? "error" : ""}` }, message);
  if (action) {
    node.append(" ", el("button", { class: "small secondary", type: "button", onclick: () => { action.run(); node.remove(); } },
      action.label));
  }
  region.append(node);
  setTimeout(() => node.remove(), action ? 8000 : 4000);
}

export function coverImg(url, alt = "", cls = "") {
  const img = el("img", { src: url || "/static/img/placeholder.svg", alt, loading: "lazy", class: cls,
    referrerpolicy: "no-referrer" });
  img.addEventListener("error", () => { if (!img.src.endsWith("/placeholder.svg")) img.src = "/static/img/placeholder.svg"; },
    { once: true });
  return img;
}

// ------------------------------------------------------------------ formatting
export const fmtEpisodes = (n) => (n === null || n === undefined ? t("common.unknown")
  : `${n} ${n === 1 ? t("common.episode") : t("common.episodes")}`);
export const fmtScore = (s) => (s === null || s === undefined ? t("common.no_score") : `★ ${Number(s).toFixed(2)}`);
export const fmtMinutes = (m) => {
  if (m === null || m === undefined) return t("common.unknown");
  const h = Math.floor(m / 60), r = Math.round(m % 60);
  return h ? `${h} h ${r} ${t("common.min")}` : `${r} ${t("common.min")}`;
};
export const animeUrl = (key) => "/anime/" + encodeURIComponent(key);

export function metaLine(item) {
  const parts = [item.type || t("common.unknown"), fmtEpisodes(item.episodes), fmtScore(item.catalog_score)];
  if (item.year) parts.push(String(item.year));
  return parts.join(" · ");
}

export function reasonsList(reasons, max = 3) {
  if (!reasons || !reasons.length) return null;
  return el("ul", { class: "reasons" }, reasons.slice(0, max).map((r) => el("li", {}, r.text)));
}

export function animeCard(item, { actions = [], badge, showReasons = true } = {}) {
  const link = animeUrl(item.item_key);
  return el("article", { class: "card", dataset: { itemKey: item.item_key } },
    el("a", { class: "cover", href: link, tabindex: "-1", "aria-hidden": "true" }, coverImg(item.cover_url, "")),
    el("div", { class: "body" },
      el("a", { class: "title", href: link }, item.title),
      el("div", { class: "meta" }, metaLine(item)),
      badge ? el("div", {}, el("span", { class: "badge accent" }, badge)) : null,
      showReasons ? reasonsList(item.reasons) : null,
      actions.length ? el("div", { class: "actions" }, actions) : null));
}

// ------------------------------------------------------------------ shared actions
let _collections = null;
export async function myCollections(force = false) {
  if (!_collections || force) _collections = (await api("/collections")).items;
  return _collections;
}

export function saveButton(item) {
  const btn = el("button", { class: "small secondary", type: "button", "aria-haspopup": "true" }, t("common.save_to"));
  btn.addEventListener("click", async () => {
    if (!loggedIn) { toast(t("common.signin_required")); return; }
    const existing = btn.parentElement.querySelector(".save-menu");
    if (existing) { existing.remove(); return; }
    let cols;
    try { cols = await myCollections(); } catch (e) { toast(errorText(e), "error"); return; }
    const menu = el("div", { class: "save-menu chips" });
    if (!cols.length) menu.append(el("a", { href: "/collections" }, t("collections.create")));
    for (const c of cols) {
      menu.append(el("button", { class: "chip", type: "button", onclick: async () => {
        try {
          await api(`/collections/${c.id}/items/${encodeURIComponent(item.item_key)}`, { method: "PUT" });
          toast(`${t("common.saved")}: ${c.name}`);
          menu.remove();
        } catch (e) { toast(errorText(e), "error"); }
      } }, c.name));
    }
    btn.after(menu);
  });
  return btn;
}

export function notInterestedButton(item, { requestId, surface, algorithm, onDone } = {}) {
  const btn = el("button", { class: "small secondary", type: "button" }, t("common.not_interested"));
  btn.addEventListener("click", async () => {
    try {
      const res = await api("/feedback", { method: "POST", body: { event_type: "not_interested", item_key: item.item_key,
        request_id: requestId || null, surface: surface || null, algorithm: algorithm || null } });
      onDone && onDone();
      toast(t("anime.not_interested_on"), "info", {
        label: t("common.undo"),
        run: async () => { await api(`/exclusions/${res.exclusion_id}`, { method: "DELETE" }); onDone && onDone(true); },
      });
    } catch (e) { toast(errorText(e), "error"); }
  });
  return btn;
}

export async function logClick(item, requestId, surface) {
  if (!loggedIn) return;
  try {
    await api("/feedback", { method: "POST", body: { event_type: "click", item_key: item.item_key, request_id: requestId,
      surface } });
  } catch (_) { /* clicks are best-effort */ }
}

export function readQuery() {
  return Object.fromEntries(new URLSearchParams(location.search).entries());
}

export function writeQuery(params) {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== "" && v !== null && v !== undefined) q.set(k, v);
  const s = q.toString();
  history.replaceState(null, "", location.pathname + (s ? "?" + s : ""));
}

// ------------------------------------------------------------------ header
document.getElementById("logout-button")?.addEventListener("click", async () => {
  try { await api("/auth/logout", { method: "POST" }); } catch (_) { /* already signed out */ }
  location.href = "/login";
});
