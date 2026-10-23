// Viewing plan: budgeted, prerequisite-aware plan (GA over a knapsack-style problem), saved revisions, export.
import { animeUrl, api, el, errorText, fmtMinutes, handleAuthError, main, readQuery, showState, t, toast } from "../common.js";

const form = document.getElementById("plan-form");
const result = document.getElementById("plan-result");
const errBox = document.getElementById("plan-error");
const slots = document.getElementById("slots");
const planId = main.dataset.planId ? Number(main.dataset.planId) : null;
let current = null;
let pending = { remove: new Set(), order: null, edits: new Map() };

function addSlot(label = "", minutes = 120) {
  const row = el("div", { class: "slot-row" },
    el("input", { type: "text", value: label, maxlength: "60", "aria-label": t("plan.slot_label"), placeholder: t("plan.slot_label") }),
    el("input", { type: "number", min: "1", max: "1440", value: String(minutes), "aria-label": t("plan.slot_minutes") }),
    el("button", { type: "button", class: "small secondary", onclick: () => row.remove() }, t("common.remove")));
  slots.append(row);
}

function selected(sel) { return [...sel.selectedOptions].map((o) => o.value); }

function request() {
  const minutes = Math.round(Number(form.hours.value || 0) * 60 + Number(form.extra_minutes.value || 0));
  return {
    budget_minutes: minutes, max_titles: Number(form.max_titles.value || 3),
    mode: form.querySelector("input[name=mode]:checked").value, diversity: form.diversity.value,
    include_genres: selected(form.include), exclude_genres: selected(form.exclude),
    min_score: form.min_score.value === "" ? null : Number(form.min_score.value),
    max_episodes_per_title: form.max_eps.value === "" ? null : Number(form.max_eps.value),
    allow_prerequisites: form.prereq.checked, break_minutes_per_title: Number(form.breaks.value || 0),
    daily_slots: [...slots.querySelectorAll(".slot-row")].map((r, i) => {
      const [label, mins] = r.querySelectorAll("input");
      return { label: label.value.trim() || `Session ${i + 1}`, minutes: Number(mins.value || 0) };
    }).filter((s) => s.minutes > 0),
    must_include: [], title: form.plan_name.value.trim() || null,
  };
}

function prefill(req) {
  if (req.budget_minutes) { form.hours.value = Math.floor(req.budget_minutes / 60); form.extra_minutes.value = req.budget_minutes % 60; }
  if (req.max_titles) form.max_titles.value = req.max_titles;
  if (req.diversity) form.diversity.value = req.diversity;
  if (req.min_score !== undefined && req.min_score !== null) form.min_score.value = req.min_score;
  if (req.max_episodes_per_title) form.max_eps.value = req.max_episodes_per_title;
  if (req.mode) form.querySelector(`input[name=mode][value=${req.mode}]`).checked = true;
  if (req.allow_prerequisites === false) form.prereq.checked = false;
  if (req.break_minutes_per_title) form.breaks.value = req.break_minutes_per_title;
  for (const g of req.include_genres || []) { const o = [...form.include.options].find((x) => x.value === g); if (o) o.selected = true; }
  for (const g of req.exclude_genres || []) { const o = [...form.exclude.options].find((x) => x.value === g); if (o) o.selected = true; }
  slots.replaceChildren();
  for (const s of req.daily_slots || []) addSlot(s.label, s.minutes);
}

function rangeText(it) {
  if (it.type === "Movie") return t("plan.full_movie");
  return `${it.episode_start}–${it.episode_end}` + (it.is_full ? "" : " *");
}

function render(plan, saved) {
  current = plan;
  const used = plan.total_minutes || 0;
  const pct = Math.min(100, (used / Math.max(1, plan.budget_minutes)) * 100);
  const bar = el("div", { class: "budget-bar", role: "img", "aria-label": `${Math.round(pct)}%` }, el("span"));
  bar.firstChild.style.width = pct + "%";
  const req = plan.request || {};
  const understood = t("plan.understood", { minutes: plan.budget_minutes, hours: Math.round(plan.budget_minutes / 6) / 10,
    titles: plan.max_titles, mode: req.mode === "partial" ? t("plan.mode.partial") : t("plan.mode.complete") });
  const feas = plan.feasibility || {};
  const status = feas.feasible ? el("p", { class: "notice ok" }, t("plan.feasible"))
    : el("div", { class: "notice warn" }, el("p", {}, t("plan.infeasible")), el("ul", {}, (feas.problems || []).map((p) => el("li", {}, p))));
  const rows = plan.items.map((it, idx) => {
    const cells = [el("td", {}, String(idx + 1)), el("td", {}, el("a", { href: animeUrl(it.item_key) }, it.title)),
      el("td", {}, rangeText(it)), el("td", {}, fmtMinutes(it.minutes)),
      el("td", {}, (it.reasons || []).slice(0, 2).map((r) => el("div", {}, r.text))),
      el("td", {}, (it.requires_titles || []).join(", ") || "–")];
    if (saved) {
      const controls = el("td", { class: "inline" },
        el("button", { type: "button", class: "small secondary", "aria-label": t("plan.move_up"), onclick: () => move(idx, -1) }, "↑"),
        el("button", { type: "button", class: "small secondary", "aria-label": t("plan.move_down"), onclick: () => move(idx, 1) }, "↓"),
        el("button", { type: "button", class: "small danger", onclick: () => { pending.remove.add(it.item_key); saveEdits(); } },
          t("common.remove")));
      if (it.type !== "Movie") {
        const end = el("input", { type: "number", min: String(it.episode_start), max: String(it.duration_evidence?.total_episodes ?? it.episode_end),
          value: String(it.episode_end), "aria-label": t("plan.edit_range") });
        end.addEventListener("change", () => { pending.edits.set(it.item_key, { item_key: it.item_key, episode_end: Number(end.value) }); saveEdits(); });
        controls.append(end);
      }
      cells.push(controls);
    }
    return el("tr", {}, cells);
  });
  const head = ["#", t("plan.col.title"), t("plan.col.range"), t("plan.col.minutes"), t("plan.col.why"), t("plan.col.first")];
  if (saved) head.push("");
  const obj = plan.objective;
  const diag = plan.diagnostics?.catalogue_not_plannable;
  result.replaceChildren(el("div", { class: "panel" },
    el("div", { class: "row-between" }, el("h2", {}, plan.title || t("plan.title")),
      saved ? savedActions(plan) : null),
    el("p", { class: "muted" }, understood),
    el("p", {}, t("plan.summary", { used: Math.round(used), budget: plan.budget_minutes, left: Math.round(plan.remaining_minutes),
      n: plan.titles })), bar, status,
    ...(plan.warnings || []).filter((w) => !(feas.problems || []).includes(w)).map((w) => el("p", { class: "notice warn" }, w)),
    plan.items.length ? el("div", { class: "table-wrap" }, el("table", {}, el("thead", {}, el("tr", {}, head.map((h) => el("th", {}, h)))),
      el("tbody", {}, rows))) : el("div", { class: "state" }, t("plan.empty")),
    plan.schedule ? scheduleBlock(plan.schedule) : null,
    el("p", { class: "muted" }, t("plan.duration_note")),
    obj || diag ? el("details", { class: "evidence" }, el("summary", {}, t("plan.how")),
      obj ? el("p", { class: "muted" }, `${obj.optimizer}: F=${obj.fitness}` + (obj.greedy_baseline ? ` · greedy F=${obj.greedy_baseline.fitness}` : "")
        + (obj.exact_oracle ? ` · ${t("plan.oracle", { gap: obj.exact_oracle.gap })}` : "")) : null,
      diag ? el("p", { class: "muted" }, t("plan.excluded", { eps: diag.unknown_episodes, dur: diag.unknown_duration, airing: diag.not_finished_airing })) : null) : null,
    saved && plan.revisions ? revisionsBlock(plan) : null));
}

function scheduleBlock(s) {
  return el("div", {}, el("h3", {}, t("plan.sessions")),
    el("ul", { class: "list" }, s.sessions.map((ss) => el("li", {}, el("strong", {}, ss.label),
      el("span", { class: "muted" }, `${Math.round(ss.used_minutes)}/${ss.capacity_minutes} ${t("common.min")}`),
      el("span", { class: "grow" }, ss.entries.map((e) => `${e.title} (${e.kind === "movie" ? "movie" : `ep ${e.episode_start}–${e.episode_end}`})`).join(", ") || "–")))),
    s.problems?.length ? el("div", { class: "notice warn" }, s.problems.map((p) => el("p", {}, p))) : null);
}

function revisionsBlock(plan) {
  return el("details", { class: "spaced" }, el("summary", {}, t("plan.revisions")),
    el("ul", {}, plan.revisions.map((r) => el("li", {}, el("a", { href: `/plans/${plan.id}?revision=${r.revision}` }, `#${r.revision}`),
      ` ${r.created_at.slice(0, 16).replace("T", " ")} · ${r.change_note} · ${r.feasible ? "✓" : "✗"}`))));
}

function savedActions(plan) {
  const exp = (fmt) => el("a", { class: "button small secondary", href: `/api/v1/plans/${plan.id}/export?format=${fmt}` }, fmt.toUpperCase());
  return el("div", { class: "inline" },
    el("button", { type: "button", class: "small", onclick: async () => {
      try { render(await api(`/plans/${plan.id}/reoptimize`, { method: "POST" }), true); toast(t("plan.reoptimize")); }
      catch (e) { toast(errorText(e), "error"); }
    } }, t("plan.reoptimize")),
    el("span", { class: "muted" }, t("plan.export") + ":"), exp("markdown"), exp("html"), exp("json"),
    el("button", { type: "button", class: "small danger", onclick: async () => {
      if (!confirm(t("common.confirm_delete"))) return;
      await api(`/plans/${plan.id}`, { method: "DELETE" });
      location.href = "/history";
    } }, t("common.delete")));
}

function move(idx, dir) {
  const keys = current.items.map((x) => x.item_key);
  const j = idx + dir;
  if (j < 0 || j >= keys.length) return;
  [keys[idx], keys[j]] = [keys[j], keys[idx]];
  pending.order = keys;
  saveEdits();
}

async function saveEdits() {
  try {
    const body = { remove_items: [...pending.remove], edit_items: [...pending.edits.values()], order: pending.order };
    pending = { remove: new Set(), order: null, edits: new Map() };
    render(await api(`/plans/${planId}`, { method: "PATCH", body }), true);
  } catch (e) { toast(errorText(e), "error"); }
}

async function previewOrSave(save) {
  errBox.hidden = true;
  showState(result, "loading");
  try {
    const req = request();
    if (save) {
      const plan = await api("/plans", { method: "POST", body: req });
      toast(t("plan.saved"));
      location.href = `/plans/${plan.id}`;
      return;
    }
    render(await api("/plans/preview", { method: "POST", body: req }), false);
  } catch (e) {
    if (handleAuthError(e)) return;
    result.replaceChildren();
    errBox.textContent = errorText(e) + (e.details?.problems ? " " + e.details.problems.join("; ") : "");
    errBox.hidden = false;
  }
}

async function init() {
  try {
    const g = await api("/tags", { query: { namespace: "genre" } });
    for (const tag of g.items) {
      form.include.append(el("option", { value: tag.value }, tag.value));
      form.exclude.append(el("option", { value: tag.value }, tag.value));
    }
  } catch (_) { /* filters optional */ }
  if (planId) {
    try {
      const rev = readQuery().revision;
      const plan = await api(`/plans/${planId}`, { query: { revision: rev } });
      prefill(plan.request || {});
      form.plan_name.value = plan.title;
      render(plan, !rev || Number(rev) === plan.current_revision);
    } catch (e) { if (!handleAuthError(e)) showState(result, "error", errorText(e)); }
    return;
  }
  const q = readQuery();
  if (q.budget) {
    prefill({ budget_minutes: Number(q.budget), max_titles: Number(q.max_titles || 3), diversity: q.diversity,
      include_genres: q.include ? q.include.split(",") : [], exclude_genres: q.exclude ? q.exclude.split(",") : [],
      min_score: q.min_score ? Number(q.min_score) : null, max_episodes_per_title: q.max_eps ? Number(q.max_eps) : null });
    previewOrSave(false);
  }
}

document.getElementById("add-slot").addEventListener("click", () => addSlot("", 120));
form.addEventListener("submit", (ev) => { ev.preventDefault(); previewOrSave(false); });
document.getElementById("save").addEventListener("click", () => previewOrSave(true));
init();
