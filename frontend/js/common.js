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
  const isFav = favs.some(f => f.id === a.id);
  const myRatings = (await api(`/api/user/${getUserId()}/ratings`)).items;
  const myScore = myRatings.find(r => r.id === a.id)?.rating || 0;

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
          <button class="btn primary" id="favBtn">${isFav ? "♥ 已收藏" : "♡ 收藏"}</button>
          <span class="chip gray">我的评分</span>${starsHTML(myScore, a.id)}
        </div>
      </div>
    </div>
    <p class="synopsis">${escapeHTML(a.synopsis)}</p>
    <div class="section-title">相似推荐（基于知识推理） <span style="font-weight:400;color:var(--muted);font-size:12px">Genre/Theme/Demographic/Studio/Length 加权</span></div>
    <div class="shuffle-row">
      <span id="simRange" style="font-size:12px;color:var(--muted)">第 1-5 名</span>
      <button class="shuffle-btn" id="shuffleBtn">换一换 ⟳</button>
    </div>
    <div id="simList" class="sim-grid"></div>
    <div class="section-title">知识图谱视角 <span style="font-weight:400;color:var(--muted);font-size:12px">Anime / Character / Studio / Franchise</span></div>
    <div id="kgView" class="kg-view">加载中…</div>`;

  $("#favBtn").onclick = () => toggleFav(a.id);
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

async function toggleFav(id) {
  const res = await api("/api/favorite", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: getUserId(), anime_id: id }),
  });
  $("#favBtn").textContent = res.favorited ? "♥ 已收藏" : "♡ 收藏";
  toast(res.favorited ? "已加入收藏，画像已更新" : "已取消收藏");
}

async function onRate(e) {
  const v = Number(e.target.dataset.v);
  if (!v) return;
  await api("/api/rate", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: getUserId(), anime_id: currentAnime.id, rating: v }),
  });
  e.currentTarget.querySelectorAll("span").forEach(s =>
    s.classList.toggle("on", Number(s.dataset.v) <= v));
  toast(`已评分 ${v} 分，推荐矩阵已更新`);
}

/* ---------- 页面公共骨架初始化 ---------- */
function initPage(active) {
  // 导航高亮 + 用户选择器
  $$(".nav-btn").forEach(btn => {
    if (btn.dataset.page === active) btn.classList.add("active");
    else btn.classList.remove("active");
  });
  const sel = $("#userSelect");
  if (sel) {
    sel.value = String(getUserId());
    sel.onchange = e => { setUserId(Number(e.target.value)); };
  }
  const modal = $("#modal");
  if (modal) modal.addEventListener("click", e => { if (e.target === modal) closeModal(); });
}
