"""AniLogic FastAPI application (single entry point: API + server-rendered pages)."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import PROJECT_ROOT, Settings, get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging, get_logger, log_event, new_request_id, request_id_var
from app.db.session import run_migrations
from app.services.serving import StateHolder

log = get_logger("http")


def _error(status: int, code: str, message: str, request_id: str, details: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message,
                                                               "details": details or {}, "request_id": request_id}})


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import anyio.to_thread

        anyio.to_thread.current_default_thread_limiter().total_tokens = max(4, int(settings.app.cpu_workers) * 4)
        run_migrations(settings.db_url)
        app.state.holder.reload()
        yield

    app = FastAPI(title="AniLogic API", version="2.0.0", lifespan=lifespan,
                  description="AniLogic: anime discovery, grounded chat and viewing plans. See docs/api.md.")
    app.state.settings = settings
    app.state.holder = StateHolder(settings)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or new_request_id()
        rid = rid[:32]
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        try:
            if request.method in ("POST", "PUT", "PATCH", "DELETE") and request.url.path.startswith("/api/"):
                origin = request.headers.get("origin")
                if origin and urlparse(origin).netloc != request.url.netloc:
                    return _error(403, "cross_origin_blocked", "Cross-origin write requests are not allowed.", rid)
            response = await call_next(request)
        except Exception:  # pragma: no cover - last resort; details only in logs
            log.exception("unhandled error")
            response = _error(500, "internal_error", "Something went wrong. Please try again.", rid)
        dur = round((time.perf_counter() - t0) * 1000, 1)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        if not request.url.path.startswith(("/docs", "/redoc", "/openapi")):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; img-src 'self' data: https://cdn.myanimelist.net; "
                "style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        log_event(log, "request", method=request.method, path=request.url.path, status=response.status_code,
                  duration_ms=dur, level=logging.INFO)
        request_id_var.reset(token)
        return response

    @app.exception_handler(AppError)
    async def app_error(request: Request, exc: AppError):
        return _error(exc.status_code, exc.code, exc.message, request_id_var.get(), exc.details)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        fields = [{"loc": [str(x) for x in e.get("loc", [])], "msg": e.get("msg", "")} for e in exc.errors()][:10]
        return _error(422, "validation_failed", "Some fields are invalid.", request_id_var.get(), {"fields": fields})

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        if request.url.path.startswith("/api/") or exc.status_code != 404:
            code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
            return _error(exc.status_code, code, str(exc.detail), request_id_var.get())
        from app.web.pages import not_found_page

        return not_found_page(request)

    from app.api.v1 import admin as admin_api
    from app.api.v1 import anime, auth, chat, collections, images, me, plans, recommendations, scenes, system

    for r in (auth, me, anime, collections, recommendations, chat, plans, scenes, images, system):
        app.include_router(r.router, prefix="/api/v1")
    app.include_router(admin_api.router, prefix="/api/v1")

    from app.web import pages

    app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "app" / "web" / "static"), name="static")
    app.include_router(pages.router)
    return app


app = create_app() if __name__ != "__main__" else None


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    s = get_settings()
    uvicorn.run("app.main:app", host=s.app.host, port=s.app.port)
