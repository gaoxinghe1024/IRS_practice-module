/* home.js — 主页：算法③核心推荐（ICF+GA Fusion） + 新剧速递（DB直读） + 观看计划（GA附加应用） */

async function loadHome() {
  const data = await api(`/api/recommend/home?user_id=${getUserId()}&top_k=5`);
  const g = data.ga;
  $("#gaInfo").innerHTML =
    `<span>算法② ICF 召回候选：<b>20 部</b></span>` +
    `<span>算法③ GA 最优适应度：<b>${g.best_fitness}</b></span>` +
    `<span>贪心基线：<b>${g.baseline_fitness}</b></span>` +
    `<span>提升：<b>+${g.improved}</b></span>` +
    `<span>权重 R/P/D/N = 0.4 / 0.3 / 0.2 / 0.1</span>`;
  $("#homeList").innerHTML = data.items.map(cardHTML).join("");
  bindCards();
}

/* 新剧速递：数据库直读 */
async function loadNew() {
  const data = await api("/api/recommend/new?limit=6");
  $("#newList").innerHTML = data.items.map(cardHTML).join("");
  $$("#newList .card").forEach(c => c.onclick = () => openModal(Number(c.dataset.id)));
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
  if (!requireAuth()) return;                    // 主界面：未登录 → 登录/注册窗口
  if (!(await checkAuth())) { location.reload(); return; }  // token 过期 → 刷新出登录门
  document.addEventListener("userchange", loadHome);
  $("#planBtn").onclick = makePlan;
  /* 偏好关键词下拉：数据库标签，选择而非输入 */
  try {
    const tags = await api("/api/tags");
    $("#planKeywords").innerHTML = `<option value="">不限</option>` +
      [...tags.genres, ...tags.themes].map(t => `<option value="${escapeHTML(t)}">${escapeHTML(t)}</option>`).join("");
  } catch (_) { /* 下拉留空不影响其他功能 */ }
  try {
    await loadHome();
    await loadNew();
  } catch (e) { toast("后端未启动：请运行 uvicorn main:app"); }
});
