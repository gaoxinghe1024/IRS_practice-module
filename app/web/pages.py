"""Server-rendered page shells. One page = one template + one page module (repo rule).

Pages are thin: all data comes from the versioned JSON API. Private pages
redirect to /login server-side when there is no valid session.
"""

from __future__ import annotations

import json
from functools import lru_cache
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.api.deps import SESSION_COOKIE
from app.core.config import PROJECT_ROOT
from app.db.session import session_factory
from app.services import auth

templates = Jinja2Templates(directory=str(PROJECT_ROOT / "app" / "web" / "templates"))
router = APIRouter(include_in_schema=False)
I18N_DIR = PROJECT_ROOT / "app" / "web" / "i18n"


@lru_cache(maxsize=4)
def strings(locale: str = "en") -> dict[str, str]:
    path = I18N_DIR / f"{locale}.json"
    if not path.exists():
        path = I18N_DIR / "en.json"
    return json.loads(path.read_text(encoding="utf-8"))


def t(key: str, locale: str = "en", **params) -> str:
    text = strings(locale).get(key, key)
    return text.format(**params) if params else text


templates.env.globals["t"] = t

NAV = [("home", "/", "nav.home"), ("discover", "/discover", "nav.discover"), ("chat", "/chat", "nav.chat"),
       ("plan", "/plan", "nav.plan"), ("collections", "/collections", "nav.collections"),
       ("profile", "/profile", "nav.profile")]


def _user(request: Request):
    db = session_factory(request.app.state.settings.db_url)()
    try:
        found = auth.resolve(db, request.cookies.get(SESSION_COOKIE))
        return found[0] if found else None
    finally:
        db.close()


def _render(request: Request, page: str, title: str, *, private: bool = True, data: dict | None = None,
            status_code: int = 200):
    user = _user(request)
    if private and user is None:
        return RedirectResponse(f"/login?next={quote(request.url.path)}", status_code=303)
    state = request.app.state.holder.get()
    settings = request.app.state.settings
    ctx = {"request": request, "page": page, "title": title, "nav": NAV, "data": data or {},
           "user": {"name": user.display_name, "is_admin": user.is_admin} if user else None,
           "app_name": settings.app.name, "tagline": settings.app.tagline, "group": settings.app.group_display,
           "dataset_mode": state.dataset_mode, "dataset_version": state.dataset_version,
           "catalog_ready": state.ready,
           "i18n_json": json.dumps(strings(settings.app.locale), ensure_ascii=False).replace("</", "<\\/")}
    return templates.TemplateResponse(request, f"{page}.html", ctx, status_code=status_code)


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return _render(request, "home", "Home")


@router.get("/login", response_class=HTMLResponse)
def login(request: Request, next: str = "/"):
    safe_next = next if next.startswith("/") and not next.startswith("//") else "/"
    if _user(request) is not None:
        return RedirectResponse(safe_next, status_code=303)
    return _render(request, "auth", "Sign in", private=False, data={"mode": "login", "next": safe_next})


@router.get("/register", response_class=HTMLResponse)
def register(request: Request):
    if _user(request) is not None:
        return RedirectResponse("/", status_code=303)
    return _render(request, "auth", "Create account", private=False, data={"mode": "register", "next": "/welcome"})


@router.get("/welcome", response_class=HTMLResponse)
def welcome(request: Request):
    return _render(request, "welcome", "Welcome")


@router.get("/discover", response_class=HTMLResponse)
def discover(request: Request):
    return _render(request, "discover", "Discover", private=False)


@router.get("/anime/{item_key}", response_class=HTMLResponse)
def anime(request: Request, item_key: str):
    return _render(request, "anime", "Anime", private=False, data={"item_key": item_key})


@router.get("/collections", response_class=HTMLResponse)
def collections(request: Request):
    return _render(request, "collections", "Collections")


@router.get("/collections/{collection_id}", response_class=HTMLResponse)
def collection_detail(request: Request, collection_id: int):
    return _render(request, "collection", "Collection", data={"collection_id": collection_id})


@router.get("/chat", response_class=HTMLResponse)
def chat(request: Request):
    return _render(request, "chat", "Chat to find")


@router.get("/chat/{session_id}", response_class=HTMLResponse)
def chat_session(request: Request, session_id: int):
    return _render(request, "chat", "Chat to find", data={"session_id": session_id})


@router.get("/plan", response_class=HTMLResponse)
def plan(request: Request):
    return _render(request, "plan", "Viewing plan")


@router.get("/plans/{plan_id}", response_class=HTMLResponse)
def plan_detail(request: Request, plan_id: int):
    return _render(request, "plan", "Viewing plan", data={"plan_id": plan_id})


@router.get("/profile", response_class=HTMLResponse)
def profile(request: Request):
    return _render(request, "profile", "Profile")


@router.get("/history", response_class=HTMLResponse)
def history(request: Request):
    return _render(request, "history", "History")


@router.get("/status", response_class=HTMLResponse)
def status(request: Request):
    return _render(request, "status", "Status", private=False)


@router.get("/dev", response_class=HTMLResponse)
def dev(request: Request):
    user = _user(request)
    if user is None or not user.is_admin:
        return not_found_page(request)
    return _render(request, "dev", "Developer tools")


def not_found_page(request: Request):
    return _render(request, "not_found", "Not found", private=False, status_code=404)
