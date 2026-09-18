/* chat.js — 算法④ 聊天找番页：RAG 语言感知搜索（对话式交互） */

const SCENE_INTROS = [
  "让我翻翻名场面库……",
  "等我想想这段在哪见过……",
  "这段描述有点印象，我查查……",
];

function addBubble(html, who, cls = "") {
  const div = document.createElement("div");
  div.className = "bubble " + who + (cls ? " " + cls : "");
  div.innerHTML = html;
  $("#chatBox").appendChild(div);
  $("#chatBox").scrollTop = $("#chatBox").scrollHeight;
  return div;
}
function renderRich(text) {
  return escapeHTML(text).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
}

/* 名场面「回忆片段」气泡：对话式呈现，与推荐位卡片刻意区分 */
function addSceneBubble(scene) {
  const kw = (scene.keywords || []).slice(0, 5)
    .map(k => `<span class="chip gray">${escapeHTML(k)}</span>`).join("");
  const goBtn = scene.anime_id
    ? `<button class="btn primary scene-go" data-id="${scene.anime_id}">去补这部番 →</button>`
    : "";
  const cover = scene.image_url
    ? `<div class="scene-cover" style="background:url('${scene.image_url}') center/cover no-repeat"></div>` : "";
  const div = addBubble(`
    <div class="scene-tag">🎞 名场面 · 《${escapeHTML(scene.title_cn || scene.title)}》第 ${scene.episode} 集</div>
    <div class="scene-flex">${cover}
      <div class="scene-main">
        <div class="scene-quote">${escapeHTML(scene.description)}</div>
        <div class="scene-kw">${kw}</div>
      </div>
    </div>
    <div class="scene-actions">${goBtn}</div>`, "bot", "scene-bubble");
  const btn = div.querySelector(".scene-go");
  if (btn) btn.onclick = () => openModal(Number(btn.dataset.id));
}

function addCandidates(animeList) {
  if (!animeList.length) return;
  const wrap = document.createElement("div");
  wrap.className = "chat-attach";
  wrap.innerHTML = `<div class="attach-label">📎 顺手从知识库捞出来的，感兴趣点开看：</div>
    <div class="attach-row">${animeList.map(a =>
      `<span class="chip attach-chip" data-id="${a.id}">${escapeHTML(a.title_cn || a.title)} ↗</span>`).join("")}</div>`;
  wrap.querySelectorAll(".attach-chip").forEach(c => c.onclick = () => openModal(Number(c.dataset.id)));
  $("#chatBox").appendChild(wrap);
  $("#chatBox").scrollTop = $("#chatBox").scrollHeight;
}

/* 打字指示：分阶段提示检索过程，强化“它在翻知识库”的对话感 */
function addTyping() {
  const stages = ["正在翻名场面库…", "正在比对知识库…", "组织语言中…"];
  const el = addBubble("正在翻名场面库…", "typing");
  let i = 0;
  el._timer = setInterval(() => {
    i = (i + 1) % stages.length;
    el.textContent = stages[i];
  }, 900);
  return el;
}

async function sendChat(message) {
  if (!message.trim()) return;
  addBubble(escapeHTML(message), "user");
  const typing = addTyping();
  const started = Date.now();
  try {
    const res = await api("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, user_id: getUserId() }),
    });
    // 保证思考提示至少出现一小会儿，更像真人在查
    const wait = Math.max(0, 700 - (Date.now() - started));
    await new Promise(r => setTimeout(r, wait));
    clearInterval(typing._timer);
    typing.remove();

    addBubble(renderRich(res.reply), "bot");
    if (res.scene_focus) addSceneBubble(res.scene_focus);
    else addCandidates(res.retrieved.anime);
  } catch (err) {
    clearInterval(typing._timer);
    typing.remove();
    addBubble("网络好像开小差了，稍后再聊一次？", "bot");
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  initPage("chat");
  if (!requireAuth()) return;
  if (!(await checkAuth())) { location.reload(); return; }
  addBubble(
    "你好呀，我是 AnimeFinder 的找番搭子 🤖 跟我聊天就行，比如：\n" +
    "· 只记得一个画面：“男主站在屋顶上和一个拿镰刀的人打架”\n" +
    "· 报得出台词：“L和夜神月对峙是哪一集”\n" +
    "· 直接开条件：“评分8.8以上、20集以内的心理战动漫”\n" +
    "我会一边翻知识库一边跟你聊，慢慢说，别客气～", "bot");
  $("#chatSend").onclick = () => { sendChat($("#chatInput").value); $("#chatInput").value = ""; };
  $("#chatInput").onkeydown = e => { if (e.key === "Enter") { sendChat($("#chatInput").value); $("#chatInput").value = ""; } };
  $$(".chat-suggest button").forEach(b => b.onclick = () => sendChat(b.dataset.q));
});
