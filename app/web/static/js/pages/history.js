// History: the signed-in user's own chats and saved plans.
import { api, el, errorText, fmtMinutes, handleAuthError, showState, t } from "../common.js";

async function load() {
  const chats = document.getElementById("chats");
  const plans = document.getElementById("plans");
  showState(chats, "loading");
  showState(plans, "loading");
  try {
    const res = await api("/me/history");
    chats.replaceChildren(...res.chats.map((c) => el("li", {}, el("a", { class: "grow", href: `/chat/${c.id}` }, c.title),
      el("small", { class: "muted" }, c.updated_at.slice(0, 16).replace("T", " ")))));
    if (!res.chats.length) showState(chats, "empty", t("history.none"));
    plans.replaceChildren(...res.plans.map((p) => el("li", {}, el("a", { class: "grow", href: `/plans/${p.id}` }, p.title),
      el("span", { class: "muted" }, `${fmtMinutes(p.summary.total_minutes)} / ${fmtMinutes(p.summary.budget_minutes)}`),
      el("small", { class: "muted" }, `rev ${p.revision}`))));
    if (!res.plans.length) showState(plans, "empty", t("history.none"));
  } catch (e) {
    if (!handleAuthError(e)) { showState(chats, "error", errorText(e), load); plans.replaceChildren(); }
  }
}
load();
