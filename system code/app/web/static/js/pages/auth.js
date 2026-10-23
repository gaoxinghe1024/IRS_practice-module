// Sign in / create account.
import { api, errorText, main } from "../common.js";

const form = document.getElementById("auth-form");
const errorBox = document.getElementById("auth-error");
const submit = document.getElementById("auth-submit");
const mode = main.dataset.mode;
const next = main.dataset.next || "/";

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  errorBox.hidden = true;
  const username = form.username.value.trim();
  const password = form.password.value;
  submit.disabled = true;
  try {
    await api(`/auth/${mode === "login" ? "login" : "register"}`, { method: "POST", body: { username, password } });
    location.href = next.startsWith("/") && !next.startsWith("//") ? next : "/";
  } catch (e) {
    errorBox.textContent = errorText(e);
    errorBox.hidden = false;
    (e.details?.field === "password" ? form.password : form.username).focus();
  } finally {
    submit.disabled = false;
  }
});
form.username.focus();
