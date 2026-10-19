// Home: personalised list (cold start -> ICF recall -> GA list optimisation) + exploration list.
import { animeCard, api, el, errorText, handleAuthError, logClick, saveButton, showState, t }
  from "../common.js";

const list = document.getElementById("home-list");
const explore = document.getElementById("explore-list");
const modeText = document.getElementById("home-mode");
const balance = document.getElementById("home-balance");
const warnings = document.getElementById("home-warnings");

async function load() {
  showState(list, "loading");
  clearNode(explore);
  try {
    const [me, res] = await Promise.all([api("/me"), api("/recommendations/home", { method: "POST", body: {} })]);
    document.getElementById("onboarding-nudge").classList.toggle("hidden", me.user.onboarding_state !== "pending");
    const cs = res.cold_start || {};
    modeText.textContent = res.mode === "warm" ? t("home.mode.warm", { n: cs.usable_ratings })
      : res.mode === "partial" ? t("home.mode.partial", { n: cs.usable_ratings, need: cs.warm_gate - cs.usable_ratings })
      : t("home.mode.cold");
    const genres = new Set(res.items.flatMap((i) => i.genres || []));
    const novel = res.items.filter((i) => (i.components?.novelty ?? 0) >= 0.6).length;
    const fresh = res.items.filter((i) => i.components?.fresh).length;
    balance.textContent = res.items.length ? t("home.balance", { genres: genres.size, novel, fresh }) : "";
    clearNode(warnings);
    for (const w of res.meta.warnings || []) warnings.append(el("p", { class: "notice warn" }, w));
    clearNode(list);
    if (!res.items.length) showState(list, "empty", t("home.empty"));
    for (const item of res.items) {
      const card = animeCard(item, {
        showReasons: false,
        actions: [saveButton(item)],
      });
      card.querySelector("a.title").addEventListener("click", () => logClick(item, res.request_id, "home"));
      list.append(card);
    }
    for (const item of res.exploration || []) {
      explore.append(animeCard(item, { showReasons: false, actions: [saveButton(item)] }));
    }
  } catch (e) {
    if (handleAuthError(e)) return;
    showState(list, "error", errorText(e), load);
  }
}

function clearNode(n) { while (n.firstChild) n.removeChild(n.firstChild); }
load();
