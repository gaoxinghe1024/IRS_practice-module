/* login.js — 注册 / 登录页逻辑 */

if (isLogged()) location.replace("/");   // 已登录直接回主界面

let mode = "login";

$$(".auth-tab").forEach(btn => {
  btn.onclick = () => {
    mode = btn.dataset.tab;
    $$(".auth-tab").forEach(b => b.classList.toggle("active", b === btn));
    $("#submitBtn").textContent = mode === "login" ? "登录" : "注册并登录";
    hideError();
  };
});

function showError(msg) {
  const el = $("#authError");
  el.textContent = msg;
  el.classList.remove("hidden");
}
function hideError() { $("#authError").classList.add("hidden"); }

$("#authForm").onsubmit = async e => {
  e.preventDefault();
  hideError();
  const username = $("#username").value.trim();
  const password = $("#password").value;
  if (!username || !password) { showError("请输入用户名和密码"); return; }

  const btn = $("#submitBtn");
  btn.disabled = true;
  btn.textContent = mode === "login" ? "登录中…" : "注册中…";
  try {
    const res = await api(`/api/auth/${mode}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    setAuth(res.token, res.user_id, res.username);
    toast(`欢迎，${res.username}！`);
    setTimeout(() => { location.href = "/"; }, 600);
  } catch (err) {
    let msg = "请求失败，请重试";
    try { msg = JSON.parse(err.message).detail || msg; } catch (_) { /* keep */ }
    showError(msg);
  } finally {
    btn.disabled = false;
    btn.textContent = mode === "login" ? "登录" : "注册并登录";
  }
};

initPage("auth");
