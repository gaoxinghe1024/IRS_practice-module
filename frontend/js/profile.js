/* profile.js — 个人中心：用户信息 / 统计 / 偏好画像 / 历史评分 */

async function loadProfile() {
  const uid = getUserId();
  const [profile, ratings] = await Promise.all([
    api(`/api/user/${uid}/profile`).catch(() => null),
    api(`/api/user/${uid}/ratings`),
  ]);

  if (!profile) {
    $("#profileCard").innerHTML = `<div class="empty">用户不存在（id=${uid}）</div>`;
    return;
  }
  const u = profile.user;
  const s = profile.stats;
  $("#profileCard").innerHTML = `
    <div class="profile-head">
      <span class="user-avatar big">${escapeHTML((u.username || "U")[0].toUpperCase())}</span>
      <div>
        <div class="profile-name">${escapeHTML(u.username)}
          ${isLogged() ? '<span class="chip">已登录</span>' : '<span class="chip gray">演示模式</span>'}
        </div>
        <div class="profile-sub">${u.created_at ? "加入于 " + u.created_at : "早期演示用户"} · 性别 ${u.gender || "Unknown"}</div>
      </div>
      ${!isLogged() ? '<a class="btn primary" style="margin-left:auto" href="/static/login.html">注册 / 登录</a>' : ""}
    </div>
    <div class="profile-stats">
      <div class="stat"><b>${s.ratings}</b><span>评分总数</span></div>
      <div class="stat"><b>${s.avg_rating}</b><span>我的平均分</span></div>
      <div class="stat"><b>${s.favorites}</b><span>收藏总数</span></div>
      <div class="stat"><b>${s.folders}</b><span>收藏夹数</span></div>
    </div>`;

  $("#genreChips").innerHTML = (profile.top_genres || []).map(g =>
    `<span class="chip">${escapeHTML(g)}</span>`).join("") ||
    `<span class="chip gray">暂无足够评分数据</span>`;

  renderHistory(ratings.items || []);
}

function renderHistory(items) {
  const box = $("#historyList");
  if (!items.length) {
    box.innerHTML = `<div class="empty">还没有评分记录。打开任意作品详情，用星级打分试试。</div>`;
    return;
  }
  box.innerHTML = items.map(r => `
    <div class="history-item" data-id="${r.id}">
      <div class="history-cover" style="${r.image_url ? `background:url('${r.image_url}') center/cover no-repeat` : "background:var(--primary-soft)"}"></div>
      <div class="history-main">
        <b>${escapeHTML(r.title_cn || r.title)}</b>
        <span class="history-meta">${r.an_type} · ${r.episodes}集 · MAL ${r.score}</span>
        <span class="history-date">${r.created_at || "早期评分"}</span>
      </div>
      <div class="history-stars">${starsStatic(r.rating)}</div>
    </div>`).join("");
  $$("#historyList .history-item").forEach(el => el.onclick = () => openModal(Number(el.dataset.id)));
}

function starsStatic(rating) {
  let h = '<div class="rate-stars static">';
  for (let i = 1; i <= 5; i++) h += `<span class="${i * 2 <= rating ? "on" : ""}">★</span>`;
  return h + `</div><span class="history-num">${rating}</span>`;
}

(async function init() {
  initPage("profile");
  if (!requireAuth()) return;
  if (!(await checkAuth())) { location.reload(); return; }
  await loadProfile();
})();
