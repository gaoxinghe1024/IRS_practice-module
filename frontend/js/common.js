/* common.js — 跨页面共享：API 封装 / 用户状态 / 卡片 / 详情弹窗组件 / 导航 */

const API = "";
const GRADIENTS = [
  "linear-gradient(135deg,#5b4fd4,#9b6bdd)", "linear-gradient(135deg,#e85d75,#f0a35e)",
  "linear-gradient(135deg,#1d9e75,#5bc4a3)", "linear-gradient(135deg,#378add,#6fc2f0)",
  "linear-gradient(135deg,#ba7517,#ef9f27)", "linear-gradient(135deg,#993556,#d4537e)",
];

function $ (sel) { return document.querySelector(sel); }
function $$ (sel) { return document.querySelectorAll(sel); }

async function api(path, opts) {
  const res = await fetch(API + path, opts);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
/* 从 api() 抛出的错误里提取后端 detail 文案 */
function errMsg(e) {
  try { return JSON.parse(e.message).detail || "请求失败，请重试"; }
  catch (_) { return "请求失败，请重试"; }
}
function toast(msg) {
  const t = $("#toast");
  if (!t) { alert(msg); return; }
  t.textContent = msg;
  t.classList.remove("hidden");
  setTimeout(() => t.classList.add("hidden"), 1800);
}
function escapeHTML(s) { return String(s).replace(/</g, "&lt;").replace(/&/g, "&amp;"); }

/* ---------- 用户状态（跨页面持久化） ---------- */
function getUserId() { return Number(localStorage.getItem("af_user") || 1); }
function setUserId(id) {
  localStorage.setItem("af_user", id);
  document.dispatchEvent(new CustomEvent("userchange", { detail: id }));
}

/* ---------- 登录态（token + 用户名 + 历史账号，支持快速切换） ---------- */
function getToken() { return localStorage.getItem("af_token") || ""; }
function getUserName() { return localStorage.getItem("af_username") || ""; }
function isLogged() { return !!getToken(); }
function getAccounts() {
  try { return JSON.parse(localStorage.getItem("af_accounts") || "[]"); }
  catch (_) { return []; }
}
function setAuth(token, userId, username) {
  localStorage.setItem("af_token", token);
  localStorage.setItem("af_user", String(userId));
  localStorage.setItem("af_username", username);
  const list = getAccounts().filter(a => a.username !== username);
  list.unshift({ username, user_id: userId, token });
  localStorage.setItem("af_accounts", JSON.stringify(list.slice(0, 8)));
}
function clearAuth() {
  localStorage.removeItem("af_token");
  localStorage.removeItem("af_username");
}

/* ---------- 封面：有真实图用真图，无则渐变占位 ---------- */
function coverStyle(a, tall) {
  const h = tall ? 210 : 130;
  if (a.image_url) {
    return `background:url('${a.image_url}') center/cover no-repeat, ${GRADIENTS[a.id % GRADIENTS.length]};height:${h}px`;
  }
  return `background:${GRADIENTS[a.id % GRADIENTS.length]};height:${h}px`;
}
function titleOf(a) { return a.title_cn || a.title; }

/* ---------- 卡片 ---------- */
function cardHTML(a) {
  const reason = a.cf_reason ? `<div class="card-reason">${a.cf_reason}</div>`
    : a.match_score ? `<div class="card-reason">画像匹配分 ${a.match_score}</div>` : "";
  return `<div class="card" data-id="${a.id}">
    <div class="cover" style="${coverStyle(a)}">${a.image_url ? "" : titleOf(a)}</div>
    <div class="card-body">
      <div class="card-title">${escapeHTML(a.title)}</div>
      <div class="card-meta"><span>${a.an_type} · ${a.episodes}集</span><span class="score-badge">${a.score}</span></div>
      ${reason}
    </div></div>`;
}
function bindCards(root = document) {
  root.querySelectorAll(".card").forEach(c => c.onclick = () => openModal(Number(c.dataset.id)));
}

/* ---------- 详情弹窗（共享组件） ---------- */
let currentAnime = null;
let simOffset = 0;

async function openModal(id) {
  simOffset = 0;
  currentAnime = await api(`/api/anime/${id}`);
  await renderModal();
  $("#modal").classList.remove("hidden");
}
function closeModal() { $("#modal").classList.add("hidden"); }

function starsHTML(rating, animeId) {
  let h = `<div class="rate-stars" data-id="${animeId}">`;
  for (let i = 1; i <= 5; i++) h += `<span data-v="${i * 2}" class="${i * 2 <= rating ? "on" : ""}">★</span>`;
  return h + "</div>";
}

async function renderModal() {
  const a = currentAnime;
  const favs = (await api(`/api/user/${getUserId()}/favorites`)).items;
  const favEntry = favs.find(f => f.id === a.id);          // 已收藏时含 folder_id
  const folders = (await api(`/api/user/${getUserId()}/folders`)).items;
  const myRatings = (await api(`/api/user/${getUserId()}/ratings`)).items;
  const myScore = myRatings.find(r => r.id === a.id)?.rating || 0;

  const folderOpts = folders.map(f =>
    `<option value="${f.id}" ${(favEntry ? f.id === favEntry.folder_id : f.id === 0) ? "selected" : ""}>${escapeHTML(f.name)}</option>`
  ).join("");

  $("#modalBody").innerHTML = `
    <div class="detail-head">
      <div class="detail-cover" style="${coverStyle(a, true)}"></div>
      <div style="flex:1">
        <div class="detail-title">${escapeHTML(titleOf(a))}</div>
        <div class="detail-title-en">${escapeHTML(a.title)} · ${a.studios} · ${a.aired} · ${a.demographics || ""}</div>
        <div class="detail-meta">
          <span class="chip">★ ${a.score}</span>
          ${a.genres.map(g => `<span class="chip">${escapeHTML(g)}</span>`).join("")}
          ${a.themes.map(t => `<span class="chip gray">${escapeHTML(t)}</span>`).join("")}
        </div>
        <div class="detail-meta"><span class="chip gray">${a.an_type} / ${a.episodes} 集</span><span class="chip gray">分级 ${a.rating_age}</span></div>
        <div class="detail-actions">
          <button class="btn primary" id="favBtn">${favEntry ? "♥ 已收藏" : "♡ 收藏"}</button>
          <select id="favFolderSel" class="fav-folder-sel" title="收藏到哪个收藏夹">
            ${folderOpts}
          </select>
          <span class="chip gray">我的评分</span>${starsHTML(myScore, a.id)}
        </div>
      </div>
    </div>
    <p class="synopsis">${escapeHTML(a.synopsis)}</p>
    <div class="section-title">相似推荐 <span class="algo-tag">算法① · Knowledge-Based Reasoning</span>
      <span style="font-weight:400;color:var(--muted);font-size:12px">Genre/Theme/Demographic/Studio/Length 五维加权</span></div>
    <div class="shuffle-row">
      <span id="simRange" style="font-size:12px;color:var(--muted)">第 1-5 名</span>
      <button class="shuffle-btn" id="shuffleBtn">换一换 ⟳</button>
    </div>
    <div id="simList" class="sim-grid"></div>
    <div class="section-title">知识图谱视角 <span style="font-weight:400;color:var(--muted);font-size:12px">Anime / Character / Studio / Franchise</span></div>
    <div id="kgView" class="kg-view">加载中…</div>`;

  $("#favBtn").onclick = () => toggleFav(a.id, favEntry);
  $("#favFolderSel").onchange = e => moveFav(a.id, favEntry, Number(e.target.value));
  $("#modalBody").querySelector(".rate-stars").onclick = onRate;
  $("#shuffleBtn").onclick = () => { simOffset += 5; loadSimilar(); };
  loadSimilar();
  loadKG(a.id);
}

async function loadSimilar() {
  const list = await api(`/api/anime/${currentAnime.id}/similar?offset=${simOffset}&limit=5`);
  $("#simRange").textContent = `第 ${simOffset + 1}-${simOffset + list.length} 名`;
  $("#simList").innerHTML = list.map(s => `
    <div class="sim-item" data-id="${s.id}">
      <b>${escapeHTML(titleOf(s))}</b>
      <span class="tags">${s.reason_tags.slice(0, 3).map(escapeHTML).join(" / ") || "综合属性相似"}</span>
      <span class="tags" style="display:block;margin-top:4px">相似度 ${(s.similarity * 100).toFixed(1)}%</span>
    </div>`).join("");
  $$("#simList .sim-item").forEach(el => el.onclick = () => openModal(Number(el.dataset.id)));
}

async function loadKG(id) {
  const kg = await api(`/api/anime/${id}/kg`);
  const animeNodes = kg.nodes.filter(n => n.type === "Anime");
  const others = kg.nodes.filter(n => n.type !== "Anime");
  let html = "";
  if (others.length) {
    html += `<div class="kg-row"><span class="rel">▾ 作品关联节点</span>` +
      others.map(n => `<span class="kg-node ${n.type}">${escapeHTML(n.label)}</span>`).join("") + `</div>`;
  }
  if (animeNodes.length > 1) {
    html += `<div class="kg-row"><span class="rel">▾ 同系列作品（belongs_to）</span>` +
      animeNodes.filter(n => n.id !== `anime-${id}`)
        .map(n => `<span class="kg-node">${escapeHTML(n.label)}</span>`).join("") + `</div>`;
  }
  $("#kgView").innerHTML = html || `<div class="kg-row"><span class="rel">暂无关联节点</span></div>`;
}

async function toggleFav(id, favEntry = null) {
  const folderId = Number($("#favFolderSel")?.value || 0);
  try {
    if (favEntry) {
      if (!confirm("取消收藏该作品？（会从所有收藏夹移除）")) {
        $("#favFolderSel").value = String(favEntry.folder_id);  // 还原选择
        return;
      }
      await api("/api/favorite", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: getToken(), anime_id: id }),
      });
      $("#favBtn").textContent = "♡ 收藏";
      toast("已取消收藏");
    } else {
      await api("/api/favorite", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: getToken(), anime_id: id, folder_id: folderId }),
      });
      $("#favBtn").textContent = "♥ 已收藏";
      toast("已收藏，画像已更新");
    }
    renderModal();  // 刷新收藏状态与文件夹选项
  } catch (e) { toast(errMsg(e)); }
}

/* 已收藏的作品换收藏夹（夹内 → 夹外/另一夹） */
async function moveFav(id, favEntry, toFolder) {
  if (!favEntry || toFolder === favEntry.folder_id) return;
  try {
    await api("/api/favorites/move", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token: getToken(), anime_id: id,
                             from_folder: favEntry.folder_id, to_folder: toFolder }),
    });
    toast("已移动到新收藏夹");
    renderModal();
  } catch (e) { toast(errMsg(e)); }
}

async function onRate(e) {
  const v = Number(e.target.dataset.v);
  if (!v) return;
  // 关键修复：事件派发结束后 e.currentTarget 会被浏览器置空，
  // 必须在 await 之前把星星容器捕获到局部变量，否则更新 UI 时拿到 null
  const starsEl = e.currentTarget;
  const animeId = currentAnime.id;
  try {
    await api("/api/rate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token: getToken(), anime_id: animeId, rating: v }),
    });
    starsEl.querySelectorAll("span").forEach(s =>
      s.classList.toggle("on", Number(s.dataset.v) <= v));
    toast(`已评分 ${v} 分，推荐矩阵已更新`);
  } catch (err) {
    const msg = errMsg(err);
    toast(msg);
    if (msg.includes("过期")) { clearAuth(); setTimeout(() => location.href = "/", 1200); }
  }
}

/* ---------- 页面公共骨架初始化 ---------- */
/* 登录卡：头像 + 用户名，点击弹出历史账号快速切换 / 登录其他账号 / 退出登录 */
function renderUserBox() {
  const box = $("#userBox");
  if (!box) return;
  if (!isLogged()) {
    box.innerHTML = `<a class="btn primary login-cta" href="/">登录 / 注册</a>`;
    return;
  }
  const name = getUserName();
  box.innerHTML = `
    <div class="user-card" id="userCard">
      <span class="user-avatar">${escapeHTML((name[0] || "U").toUpperCase())}</span>
      <div class="user-card-info"><b>${escapeHTML(name)}</b><span>当前登录</span></div>
      <span class="user-card-arrow">▾</span>
    </div>
    <div class="user-pop hidden" id="userPop"></div>`;
  const card = $("#userCard"), pop = $("#userPop");
  card.onclick = e => { e.stopPropagation(); renderUserPop(); pop.classList.toggle("hidden"); };
  document.addEventListener("click", e => {
    if (!pop.contains(e.target)) pop.classList.add("hidden");
  });
}

function renderUserPop() {
  const pop = $("#userPop");
  if (!pop) return;
  const cur = getUserName();
  const others = getAccounts().filter(a => a.username !== cur);
  pop.innerHTML = `
    <div class="pop-label">快速切换账号</div>
    ${others.length ? others.map(a => `
      <button class="pop-item" data-u="${escapeHTML(a.username)}" data-i="${a.user_id}" data-t="${a.token}">
        <span class="user-avatar sm">${escapeHTML((a.username[0] || "U").toUpperCase())}</span>${escapeHTML(a.username)}
      </button>`).join("")
      : `<div class="pop-empty">暂无其他登录过的账号</div>`}
    <div class="pop-sep"></div>
    <button class="pop-item" id="popLogin">＋ 登录其他账号</button>
    <button class="pop-item danger" id="popLogout">⎋ 退出登录</button>`;
  pop.querySelectorAll(".pop-item[data-u]").forEach(btn => btn.onclick = () => switchAccount(btn.dataset.t));
  $("#popLogin").onclick = () => { clearAuth(); location.href = "/"; };
  $("#popLogout").onclick = doLogout;
}

async function switchAccount(token) {
  try {
    const me = await api(`/api/auth/me?token=${encodeURIComponent(token)}`);
    localStorage.setItem("af_token", token);
    localStorage.setItem("af_user", String(me.id));
    localStorage.setItem("af_username", me.username);
    location.reload();
  } catch (_) {
    localStorage.setItem("af_accounts", JSON.stringify(
      getAccounts().filter(a => a.token !== token)));
    toast("该账号登录已过期，请重新登录");
    renderUserPop();
  }
}

async function doLogout() {
  await api("/api/auth/logout", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token: getToken() }),
  }).catch(() => {});
  clearAuth();
  location.href = "/";
}

/* 主界面登录门：未登录时把整页变成登录/注册窗口 */
function requireAuth() {
  if (isLogged()) return true;
  const gate = document.createElement("div");
  gate.className = "auth-gate";
  gate.innerHTML = `
    <div class="gate-card">
      <div class="gate-logo"><span class="logo-icon">アニ</span><h2>AnimeFinder</h2>
        <p>登录后开始你的专属找番之旅</p></div>
      <div class="gate-tabs">
        <button class="gate-tab active" data-m="login">登录</button>
        <button class="gate-tab" data-m="register">注册</button>
      </div>
      <form id="gateForm">
        <input id="gateUser" placeholder="用户名（2-20 字符）" autocomplete="username">
        <input id="gatePass" type="password" placeholder="密码（至少 6 位）" autocomplete="current-password">
        <div class="gate-err hidden" id="gateErr"></div>
        <button class="btn primary gate-submit" id="gateSubmit">登录</button>
      </form>
    </div>`;
  document.body.appendChild(gate);
  let mode = "login";
  gate.querySelectorAll(".gate-tab").forEach(t => t.onclick = () => {
    mode = t.dataset.m;
    gate.querySelectorAll(".gate-tab").forEach(b => b.classList.toggle("active", b === t));
    $("#gateSubmit").textContent = mode === "login" ? "登录" : "注册并登录";
    $("#gateErr").classList.add("hidden");
  });
  $("#gateForm").onsubmit = async ev => {
    ev.preventDefault();
    const username = $("#gateUser").value.trim(), password = $("#gatePass").value;
    const err = $("#gateErr");
    err.classList.add("hidden");
    try {
      const res = await api(`/api/auth/${mode}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      setAuth(res.token, res.user_id, res.username);
      location.reload();
    } catch (e) {
      err.textContent = errMsg(e);
      err.classList.remove("hidden");
    }
  };
  return false;
}

/* token 有效性校验：过期则清除并回登录门 */
async function checkAuth() {
  if (!isLogged()) return false;
  try { await api(`/api/auth/me?token=${encodeURIComponent(getToken())}`); return true; }
  catch (_) { clearAuth(); return false; }
}

function initPage(active) {
  // 导航高亮 + 用户区
  $$(".nav-btn").forEach(btn => {
    if (btn.dataset.page === active) btn.classList.add("active");
    else btn.classList.remove("active");
  });
  renderUserBox();
  const modal = $("#modal");
  if (modal) modal.addEventListener("click", e => { if (e.target === modal) closeModal(); });
}
