// Chat-to-find (algorithm 4): multi-turn constraints, grounded answers with sources, local mode.
import { animeUrl, api, coverImg, el, errorText, handleAuthError, main, metaLine, reasonsList, saveButton, showState, t,
  toast } from "../common.js";

const list = document.getElementById("session-list");
const messages = document.getElementById("messages");
const form = document.getElementById("composer");
const input = document.getElementById("message");
const send = document.getElementById("send");
let sessionId = main.dataset.sessionId ? Number(main.dataset.sessionId) : null;
let lastState = null;

async function loadSessions() {
  try {
    const res = await api("/chat/sessions");
    list.replaceChildren(...res.items.map((s) => el("a", { href: `/chat/${s.id}`, "aria-current": String(s.id === sessionId) },
      s.title)));
  } catch (e) { handleAuthError(e); }
}

function chips(items) {
  if (!items?.length) return null;
  return el("div", {}, el("small", { class: "muted" }, t("chat.conditions") + ": "),
    el("span", { class: "chips" }, items.map((c) => el("span", { class: `chip ${c.kind}`,
      title: c.kind === "hard" ? t("chat.hard") : t("chat.soft") }, c.label))));
}

function evidenceBlock(ev, spoiler) {
  if (!ev?.length) return null;
  const details = el("details", { class: "evidence" }, el("summary", {}, `${t("chat.evidence")} (${ev.length})`),
    el("ul", {}, ev.map((d) => el("li", {}, el("strong", {}, `[${d.doc_id}] `), `${d.title} · ${d.kind} · ${d.source_type}`,
      el("div", { class: "muted" }, d.snippet), el("small", { class: "muted" }, d.source_ref)))));
  if (!spoiler) return details;
  const wrap = el("div", {}, el("p", { class: "notice warn" }, t("chat.spoiler"), " ",
    el("button", { type: "button", class: "small secondary", onclick: (ev2) => {
      ev2.target.closest(".notice").remove();
      details.classList.remove("hidden");
      details.open = true;
    } }, t("chat.show"))), details);
  details.classList.add("hidden");
  return wrap;
}

function planLink(plan) {
  if (!plan) return null;
  const r = plan.request || {};
  const q = new URLSearchParams({ budget: r.budget_minutes, max_titles: r.max_titles, diversity: r.diversity || "normal" });
  if (r.include_genres?.length) q.set("include", r.include_genres.join(","));
  if (r.exclude_genres?.length) q.set("exclude", r.exclude_genres.join(","));
  if (r.min_score) q.set("min_score", r.min_score);
  if (r.max_episodes_per_title) q.set("max_eps", r.max_episodes_per_title);
  return el("a", { class: "button small", href: "/plan?" + q.toString() }, t("chat.open_plan"));
}

function relaxButtons(options) {
  const numeric = (options || []).filter((o) => "max_episodes" in o.change || "min_score" in o.change);
  if (!numeric.length) return null;
  return el("div", { class: "inline" }, el("small", { class: "muted" }, t("chat.relax")),
    numeric.map((o) => el("button", { type: "button", class: "small secondary", onclick: () => {
      const c = o.change;
      const text = "max_episodes" in c && c.max_episodes ? `at most ${c.max_episodes} episodes`
        : "min_score" in c && c.min_score ? `score at least ${c.min_score}` : null;
      if (text) sendMessage(text);
    } }, o.reason)));
}

function renderAssistant(p) {
  const extras = el("div", { class: "extras" });
  const wrap = el("div", { class: "msg assistant" }, el("div", { class: "bubble" }, p.answer), extras);
  extras.append(chips(p.chips) || "");
  if (p.answer_mode === "local_template") extras.append(el("small", { class: "muted" }, t("common.local_mode")));
  for (const w of p.warnings || []) extras.append(el("small", { class: "muted" }, w));
  if (p.unknowns?.length) extras.append(el("small", {}, t("chat.unknowns", { items: p.unknowns.join(", ") })));
  if (p.items?.length) {
    extras.append(el("ul", { class: "result-rows" }, p.items.map((it) => el("li", { class: "result-row" },
      coverImg(it.cover_url, "", "thumb"),
      el("div", {}, el("a", { href: animeUrl(it.item_key) }, el("strong", {}, it.title)),
        el("div", { class: "muted" }, metaLine(it)), reasonsList(it.reasons, 2)),
      saveButton(it)))));
  }
  const rb = relaxButtons(p.relaxation_options);
  if (rb) extras.append(rb);
  const pl = planLink(p.plan_preview);
  if (pl) extras.append(pl);
  const ev = evidenceBlock(p.evidence, p.spoiler);
  if (ev) extras.append(ev);
  if (p.chips?.some((c) => c.kind === "hard")) {
    extras.append(el("button", { type: "button", class: "small secondary", onclick: savePrefs }, t("chat.save_prefs")));
  }
  return wrap;
}

function renderUser(text) {
  return el("div", { class: "msg user" }, el("div", { class: "bubble" }, text));
}

async function savePrefs() {
  try {
    const s = await api(`/chat/sessions/${sessionId}`);
    const me = await api("/me");
    const liked = [...me.preferences.liked_tags];
    for (const [namespace, value] of s.state.include_tags || []) {
      if (!liked.some((x) => x.namespace === namespace && x.value === value)) liked.push({ namespace, value });
    }
    await api("/me/preferences", { method: "PATCH", body: { liked_tags: liked } });
    for (const [namespace, value] of s.state.exclude_tags || []) {
      await api("/exclusions", { method: "POST", body: { kind: "tag", tag: { namespace, value } } });
    }
    toast(t("chat.saved_prefs"));
  } catch (e) { toast(errorText(e), "error"); }
}

async function ensureSession() {
  if (sessionId) return sessionId;
  const s = await api("/chat/sessions", { method: "POST", body: {} });
  sessionId = s.id;
  history.replaceState(null, "", `/chat/${s.id}`);
  return sessionId;
}

async function sendMessage(text) {
  text = text.trim();
  if (!text) return;
  send.disabled = true;
  messages.append(renderUser(text));
  const pending = el("div", { class: "state loading" }, t("chat.thinking"));
  messages.append(pending);
  pending.scrollIntoView({ block: "nearest" });
  try {
    await ensureSession();
    const res = await api(`/chat/sessions/${sessionId}/messages`, { method: "POST", body: { message: text } });
    pending.replaceWith(renderAssistant(res));
    lastState = res;
    loadSessions();
  } catch (e) {
    if (handleAuthError(e)) return;
    pending.replaceWith(el("div", { class: "state error", role: "alert" }, errorText(e)));
  } finally {
    send.disabled = false;
    input.focus();
  }
}

async function loadHistory() {
  if (!sessionId) return;
  showState(messages, "loading");
  try {
    const s = await api(`/chat/sessions/${sessionId}`);
    messages.replaceChildren();
    for (const m of s.messages) messages.append(m.role === "user" ? renderUser(m.content) : renderAssistant(m.payload));
  } catch (e) {
    if (!handleAuthError(e)) showState(messages, "error", errorText(e));
  }
}

async function setupExamples() {
  let title = "Cowboy Bebop";
  try {
    const top = await api("/anime", { query: { sort: "popularity", page_size: 1 } });
    if (top.items[0]) title = top.items[0].title;
  } catch (_) { /* keep default */ }
  const box = document.getElementById("examples");
  for (const k of ["chat.example.1", "chat.example.2", "chat.example.3", "chat.example.4"]) {
    const text = t(k, { title });
    box.append(el("button", { type: "button", class: "chip", onclick: () => sendMessage(text) }, text));
  }
}

async function setupCapabilities() {
  try {
    const st = await api("/system/status");
    const notes = document.getElementById("capability-notes");
    const llm = st.capabilities.llm;
    if (llm && llm.status !== "ready") notes.append(el("p", { class: "notice" }, llm.detail));
    const shot = st.capabilities.screenshot;
    if (shot && (shot.status === "ready" || shot.status === "degraded")) {
      document.getElementById("image-box").classList.remove("hidden");
      if (shot.status === "degraded") document.getElementById("image-box").prepend(el("p", { class: "notice warn" }, shot.detail));
    }
  } catch (_) { /* status is informational */ }
}

document.getElementById("image-input").addEventListener("change", async (ev) => {
  const file = ev.target.files[0];
  const out = document.getElementById("image-results");
  if (!file) return;
  showState(out, "loading");
  const fd = new FormData();
  fd.append("file", file);
  try {
    const res = await api("/images/identify", { method: "POST", form: fd });
    out.replaceChildren(el("p", { class: "muted" }, t("chat.image_results")), el("p", { class: "muted" }, res.note),
      el("ul", { class: "list" }, res.candidates.map((c) => el("li", {}, el("a", { href: animeUrl(c.item_key) }, c.title),
        el("span", { class: "badge" }, `sim ${c.similarity}`),
        el("small", { class: "muted" }, c.frames.map((f) => (f.episode ? `ep ${f.episode}` : "episode unknown")).join(", "))))));
  } catch (e) { showState(out, "error", errorText(e)); }
});

form.addEventListener("submit", (ev) => { ev.preventDefault(); const v = input.value; input.value = ""; sendMessage(v); });
input.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); form.requestSubmit(); }
});
document.getElementById("new-chat").addEventListener("click", () => { location.href = "/chat"; });
loadSessions();
loadHistory();
setupExamples();
setupCapabilities();
