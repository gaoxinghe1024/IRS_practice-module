/* favorites.js — 我的收藏页：收藏夹列表 ⇄ 收藏夹详情（内含作品 + 混合推荐） */

const folderListView = $("#folderListView");
const folderDetailView = $("#folderDetailView");
let currentFolderId = 0;

initPage("favorites");

$("#backToFolders").onclick = () => {
  history.replaceState(null, "", "/static/favorites.html");
  showFolderList();
};
$("#newFolderBtn").onclick = async () => {
  const name = prompt("给新收藏夹起个名字：");
  if (!name || !name.trim()) return;
  await api("/api/folders", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: getUserId(), token: getToken(), name: name.trim() }),
  });
  toast("收藏夹已创建");
  loadFolders();
};

/* ---------- 视图 A：收藏夹列表 ---------- */
async function showFolderList() {
  folderListView.classList.remove("hidden");
  folderDetailView.classList.add("hidden");
  await loadFolders();
}

/* ---------- 标签下拉：从数据库拉全库 genre/theme（标签应选择而非输入） ---------- */
async function initTagSelect() {
  const sel = $("#fusionKeywords");
  if (!sel) return;
  try {
    const tags = await api("/api/tags");
    sel.innerHTML = `<option value="">不限标签</option>` +
      `<optgroup label="类型 genre">${tags.genres.map(g => `<option value="${escapeHTML(g)}">${escapeHTML(g)}</option>`).join("")}</optgroup>` +
      `<optgroup label="主题 theme">${tags.themes.map(t => `<option value="${escapeHTML(t)}">${escapeHTML(t)}</option>`).join("")}</optgroup>`;
  } catch (_) {
    sel.innerHTML = `<option value="">不限标签</option>`;
  }
}

/* ---------- 收藏夹间移动：条目卡片上的「移动到…」下拉 ---------- */
function bindMoveSelects(items) {
  $$("#folderItems .fav-move").forEach(sel => {
    sel.onclick = e => e.stopPropagation();
    const aid = Number(sel.dataset.id);
    sel.onchange = async () => {
      const to = Number(sel.value);
      if (!sel.value) return;
      await api("/api/favorites/move", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: getToken(), anime_id: aid,
                               from_folder: currentFolderId, to_folder: to }),
      });
      toast("已移动到新收藏夹");
      openFolder(currentFolderId);
    };
  });
}

async function loadFolders() {
  const uid = getUserId();
  const data = await api(`/api/user/${uid}/folders`);
  const grid = $("#folderGrid");
  if (!data.items.length || data.items.every(f => !f.count)) {
    grid.innerHTML = `<div class="empty">还没有收藏。去「浏览全部」或详情弹窗里点 ♡ 收藏吧。</div>`;
    return;
  }
  grid.innerHTML = data.items.map(f => `
    <div class="folder-card" data-id="${f.id}" data-name="${escapeHTML(f.name)}">
      <div class="folder-cover" style="${f.cover ? `background:url('${f.cover}') center/cover no-repeat` : ""}">
        <span class="folder-count">${f.count} 部</span>
      </div>
      <div class="folder-body">
        <div class="folder-name">${escapeHTML(f.name)}</div>
        <div class="folder-meta">${f.id === 0 ? "收藏时未分夹的默认去处" : "自定义收藏夹"}</div>
      </div>
    </div>`).join("");
  $$("#folderGrid .folder-card").forEach(card =>
    card.onclick = () => openFolder(Number(card.dataset.id), card.dataset.name));
}

/* ---------- 视图 B：收藏夹详情 ---------- */
async function openFolder(folderId, name = "") {
  currentFolderId = folderId;
  history.replaceState(null, "", `/static/favorites.html?folder=${folderId}`);
  folderListView.classList.add("hidden");
  folderDetailView.classList.remove("hidden");

  // 缓存收藏夹列表，供条目「移动到…」下拉使用
  try { window.__myFolders = (await api(`/api/user/${getUserId()}/folders`)).items; }
  catch (_) { window.__myFolders = []; }

  const data = await api(`/api/folders/${folderId}?user_id=${getUserId()}&token=${encodeURIComponent(getToken())}`);
  $("#folderTitle").textContent = data.folder.name || name;
  $("#folderSub").textContent =
    `${data.items.length} 部作品 · ${data.folder.created_at ? "创建于 " + data.folder.created_at : "系统默认收藏夹"}`;
  $("#delFolderBtn").classList.toggle("hidden", folderId === 0);
  $("#delFolderBtn").onclick = async () => {
    if (!confirm(`确定删除「${data.folder.name}」？里面的作品会移回默认收藏夹。`)) return;
    await api(`/api/folders/${folderId}?token=${encodeURIComponent(getToken())}&user_id=${getUserId()}`,
      { method: "DELETE" });
    toast("收藏夹已删除");
    showFolderList();
  };

  renderFolderItems(data.items);
  renderHybrid(folderId);
}

/* 规则筛选按钮：用当前输入重新触发算法② */
$("#fusionBtn").onclick = () => renderHybrid(currentFolderId);

function renderFolderItems(items) {
  const box = $("#folderItems");
  if (!items.length) {
    box.innerHTML = `<div class="empty">这个收藏夹还是空的。</div>`;
    return;
  }
  box.innerHTML = items.map(a => `
    <div class="card" data-id="${a.id}">
      <div class="cover" style="${coverStyle(a)}">${a.image_url ? "" : titleOf(a)}
        <button class="fav-remove" data-id="${a.id}" title="从收藏夹移除">×</button>
      </div>
      <div class="card-body">
        <div class="card-title">${escapeHTML(a.title)}</div>
        <div class="card-meta"><span>${a.an_type} · ${a.episodes}集</span><span class="score-badge">${a.score}</span></div>
        <select class="fav-move" data-id="${a.id}" title="移动到其他收藏夹">
          <option value="">移动到…</option>
          <option value="0" ${currentFolderId === 0 ? "disabled" : ""}>默认收藏夹</option>
          ${(window.__myFolders || []).filter(f => f.id !== currentFolderId)
              .map(f => `<option value="${f.id}">${escapeHTML(f.name)}</option>`).join("")}
        </select>
      </div>
    </div>`).join("");
  bindCards(box);
  bindMoveSelects(items);
  box.querySelectorAll(".fav-remove").forEach(btn => {
    btn.onclick = async e => {
      e.stopPropagation();
      await api("/api/favorites/remove", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: getUserId(), token: getToken(),
                               anime_id: Number(btn.dataset.id), folder_id: currentFolderId }),
      });
      toast("已从收藏夹移除");
      openFolder(currentFolderId);
    };
  });
}

/* ---------- 算法② 混合推理（规则筛选 × 收藏夹知识画像） ---------- */
async function renderHybrid(folderId) {
  const info = $("#hybridInfo");
  const list = $("#hybridList");
  const keywords = $("#fusionKeywords").value.trim();
  const minScore = Number($("#fusionMinScore").value) || 0;
  const maxEp = Number($("#fusionMaxEp").value) || 0;
  const qs = `keywords=${encodeURIComponent(keywords)}&min_score=${minScore}&max_episodes=${maxEp}`;
  info.textContent = "混合推理计算中…";
  list.innerHTML = "";

  const data = await api(`/api/folders/${folderId}/recommend?user_id=${getUserId()}&token=${encodeURIComponent(getToken())}&${qs}`);
  if (data.message) {
    info.innerHTML = `<span>${escapeHTML(data.message)}</span>`;
    return;
  }
  const rules = data.rules_applied || {};
  const ruleTxt = [
    rules.keywords?.length ? `关键词 ${rules.keywords.join("/")}` : "",
    rules.min_score ? `评分 ≥ ${rules.min_score}` : "",
    rules.max_episodes ? `${rules.max_episodes} 集以内` : "",
  ].filter(Boolean).join(" · ");
  info.innerHTML = `
    <span>规则层：<b>${ruleTxt || "未启用（可在上方输入条件）"}</b></span>
    <span>知识层画像：<b>${(data.profile_tags || []).slice(0, 4).map(escapeHTML).join("、") || "—"}</b></span>
    ${data.relaxed ? '<span style="color:#c23a52">⚠ 规则筛空，已降级为纯知识画像排序</span>' : ""}`;
  list.innerHTML = data.items.map(a => `
    <div class="card" data-id="${a.id}">
      <div class="cover" style="${coverStyle(a)}">${a.image_url ? "" : titleOf(a)}</div>
      <div class="card-body">
        <div class="card-title">${escapeHTML(a.title)}</div>
        <div class="card-meta"><span>${a.an_type} · ${a.episodes}集</span><span class="score-badge">${a.score}</span></div>
        <div class="card-reason">${escapeHTML(a.fusion_reason || "")} · 匹配分 ${a.match_score}</div>
      </div>
    </div>`).join("") || `<div class="empty">没有符合条件的作品，试着放宽规则～</div>`;
  bindCards(list);
}

/* ---------- 入口：支持 ?folder=id 直达详情 ---------- */
(async function init() {
  if (!requireAuth()) return;
  if (!(await checkAuth())) { location.reload(); return; }
  await initTagSelect();
  const params = new URLSearchParams(location.search);
  const fid = Number(params.get("folder"));
  if (fid) {
    await openFolder(fid);
  } else {
    await showFolderList();
  }
})();
