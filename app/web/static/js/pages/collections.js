// Collections: list / create / rename / delete (items are managed on detail pages).
import { api, coverImg, el, errorText, handleAuthError, showState, t, toast } from "../common.js";

const list = document.getElementById("collection-list");
const form = document.getElementById("create-form");
const err = document.getElementById("create-error");

async function load() {
  showState(list, "loading");
  try {
    const res = await api("/collections");
    list.replaceChildren();
    if (!res.items.length) { showState(list, "empty", t("collections.empty")); return; }
    for (const c of res.items) list.append(row(c));
  } catch (e) {
    if (!handleAuthError(e)) showState(list, "error", errorText(e), load);
  }
}

function row(c) {
  const li = el("li", {}, coverImg(c.cover_url, "", "thumb"),
    el("div", { class: "grow" }, el("a", { href: `/collections/${c.id}` }, el("strong", {}, c.name)),
      el("div", { class: "muted" }, t("collections.count", { n: c.item_count }) + (c.description ? " · " + c.description : ""))));
  const rename = el("button", { type: "button", class: "small secondary", onclick: async () => {
    const name = prompt(t("collections.name"), c.name);
    if (!name || name.trim() === c.name) return;
    try { await api(`/collections/${c.id}`, { method: "PATCH", body: { name: name.trim() } }); await load(); }
    catch (e) { toast(errorText(e), "error"); }
  } }, t("common.rename"));
  const del = el("button", { type: "button", class: "small danger", onclick: async () => {
    if (!confirm(`${t("common.confirm_delete")}\n${c.name}`)) return;
    try { await api(`/collections/${c.id}`, { method: "DELETE" }); await load(); }
    catch (e) { toast(errorText(e), "error"); }
  } }, t("common.delete"));
  li.append(rename, del);
  return li;
}

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  err.hidden = true;
  try {
    await api("/collections", { method: "POST", body: { name: form.name.value.trim(), description: form.description.value.trim() } });
    form.reset();
    await load();
  } catch (e) { err.textContent = errorText(e); err.hidden = false; }
});
load();
