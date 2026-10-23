// Status: dataset mode/version and plain-language capability states.
import { api, el, errorText, showState, t } from "../common.js";

const root = document.getElementById("status-root");
const BADGE = { ready: "ok", degraded: "warn", limited_by_data: "warn", limited_by_credentials: "warn",
  not_configured: "", missing_data: "danger", artifact_mismatch: "danger", failed: "danger" };

async function load() {
  showState(root, "loading");
  try {
    const s = await api("/system/status");
    const rows = Object.values(s.capabilities).map((c) => el("tr", {}, el("td", {}, c.name.replaceAll("_", " ")),
      el("td", {}, el("span", { class: `badge ${BADGE[c.status] ?? ""}` }, c.status.replaceAll("_", " "))), el("td", {}, c.detail)));
    const models = Object.entries(s.models || {}).map(([k, v]) => el("li", {}, `${k}: ${v.status}${v.id ? " (" + v.id + ")" : ""}`));
    root.replaceChildren(
      el("p", {}, `${t("common.dataset")}: ${s.dataset_mode} · ${s.dataset_version ?? "–"}`),
      el("div", { class: "panel table-wrap" }, el("table", {}, el("thead", {}, el("tr", {},
        el("th", {}, t("status.capability")), el("th", {}, t("status.state")), el("th", {}, t("status.detail")))), el("tbody", {}, rows))),
      el("section", { class: "panel" }, el("h2", {}, t("status.models")), el("ul", {}, models)));
  } catch (e) { showState(root, "error", errorText(e), load); }
}
load();
