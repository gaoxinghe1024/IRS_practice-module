/* home.js — 主页推荐页 + 观看计划优化面板 */

async function loadHome() {
  const data = await api(`/api/recommend/home?user_id=${getUserId()}&top_k=5`);
  const g = data.ga;
  $("#gaInfo").innerHTML =
    `<span>ICF 召回候选：<b>20 部</b></span>` +
    `<span>GA 最优适应度：<b>${g.best_fitness}</b></span>` +
    `<span>贪心基线：<b>${g.baseline_fitness}</b></span>` +
    `<span>提升：<b>+${g.improved}</b></span>` +
    `<span>权重 R/P/D/N = 0.4 / 0.3 / 0.2 / 0.1</span>`;
  $("#homeList").innerHTML = data.items.map(cardHTML).join("");
  bindCards();
}

/* ---------- 观看计划 ---------- */
async function makePlan() {
  const hours = Number($("#planHours").value) || 15;
  const keywords = $("#planKeywords").value.trim();
  $("#planResult").innerHTML = `<div class="plan-loading">遗传算法寻优中…（种群 40 × 80 代）</div>`;
  try {
    const plan = await api("/api/watchplan", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: getUserId(), hours, keywords }),
    });
    const pct = Math.min(100, Math.round(plan.total_minutes / (plan.budget_hours * 60) * 100));
    $("#planResult").innerHTML = `
      <div class="plan-summary">
        <span>计划适应度：<b>${plan.plan_fitness}</b></span>
        <span>总时长：<b>${plan.total_hours} 小时</b> / 预算 ${plan.budget_hours} 小时</span>
        <span>共 <b>${plan.items.length}</b> 部</span>
        <div class="plan-bar"><div class="plan-bar-fill" style="width:${pct}%"></div></div>
      </div>
      <div class="plan-list">
        ${plan.items.map((it, i) => `
          <div class="plan-item" data-id="${it.id}">
            <span class="plan-idx">${i + 1}</span>
            <span class="plan-name">${escapeHTML(it.title_cn || it.title)}</span>
            <span class="plan-meta">${it.an_type} · ${it.episodes}集 · 约 ${(it.minutes / 60).toFixed(1)} 小时</span>
            <span class="score-badge">${it.score}</span>
          </div>`).join("")}
      </div>`;
      $$("#planResult .plan-item").forEach(el => el.onclick = () => openModal(Number(el.dataset.id)));
  } catch (e) {
    $("#planResult").innerHTML = `<div class="plan-loading">生成失败，请稍后再试</div>`;
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  initPage("home");
  document.addEventListener("userchange", loadHome);
  $("#planBtn").onclick = makePlan;
  try { await loadHome(); } catch (e) { toast("后端未启动：请运行 uvicorn main:app"); }
});
