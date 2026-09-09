/* chat.js — 聊天找番页 */

function addBubble(html, who) {
  const div = document.createElement("div");
  div.className = "bubble " + who;
  div.innerHTML = html;
  $("#chatBox").appendChild(div);
  $("#chatBox").scrollTop = $("#chatBox").scrollHeight;
  return div;
}
function renderRich(text) {
  return escapeHTML(text).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
}

async function sendChat(message) {
  if (!message.trim()) return;
  addBubble(escapeHTML(message), "user");
  const typing = addBubble("正在检索知识库…", "typing");
  try {
    const res = await api("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, user_id: getUserId() }),
    });
    typing.remove();
    addBubble(renderRich(res.reply), "bot");
    if (res.retrieved.anime.length) {
      const row = document.createElement("div");
      row.style.cssText = "align-self:flex-start;display:flex;gap:8px;flex-wrap:wrap";
      row.innerHTML = res.retrieved.anime.map(a =>
        `<span class="chip" style="cursor:pointer" data-id="${a.id}">${escapeHTML(a.title_cn || a.title)} ↗</span>`).join("");
      row.querySelectorAll(".chip").forEach(c => c.onclick = () => openModal(Number(c.dataset.id)));
      $("#chatBox").appendChild(row);
      $("#chatBox").scrollTop = $("#chatBox").scrollHeight;
    }
  } catch (err) {
    typing.remove();
    addBubble("出错了，请稍后再试。", "bot");
  }
}

window.addEventListener("DOMContentLoaded", () => {
  initPage("chat");
  addBubble("你好呀！我是 AnimeFinder 的找番助手 🤖\n你可以：\n· 用自然语言描述需求（“类似进击的巨人但短一点的”）\n· 描述记得的名场面（“L和夜神月对峙是哪一集”）\n· 直接说硬条件（“9分以上、20集以内的心理战”）", "bot");
  $("#chatSend").onclick = () => { sendChat($("#chatInput").value); $("#chatInput").value = ""; };
  $("#chatInput").onkeydown = e => { if (e.key === "Enter") { sendChat($("#chatInput").value); $("#chatInput").value = ""; } };
  $$(".chat-suggest button").forEach(b => b.onclick = () => sendChat(b.dataset.q));
});
