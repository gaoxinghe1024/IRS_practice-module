"""Layered configuration: configs/base.yaml -> configs/<mode>.yaml -> .env -> environment.

Environment overrides use ``ANILOGIC_<SECTION>__<KEY>`` (double underscore for
nesting), e.g. ``ANILOGIC_RECOMMENDATION__HOME__RERANKER=greedy``. The dataset
mode is selected with ``ANILOGIC_DATASET_MODE`` (synthetic | real). Secrets are
read only from the environment (``ANILOGIC_LLM_API_KEY``) and never persisted.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
ENV_PREFIX = "ANILOGIC_"


class _Section(BaseModel):
    model_config = ConfigDict(extra="allow")


class AppSection(_Section):
    name: str = "AniLogic"
    tagline: str = "Your ACG Friend"
    group_display: str = "Group 10"
    locale: str = "en"
    host: str = "127.0.0.1"
    port: int = 8000
    secure_cookies: bool = False
    session_ttl_hours: int = 168
    max_upload_bytes: int = 5 * 1024 * 1024
    max_upload_pixels: int = 25_000_000
    log_chat_text: bool = False
    cpu_workers: int = 4


class DatasetSection(_Section):
    mode: Literal["synthetic", "real"] = "synthetic"


class PathsSection(_Section):
    data_root: str = "data"
    artifacts_root: str = "artifacts"
    var_root: str = "var"
    evaluation_runs: str = "evaluation/runs"


class ContentRules(_Section):
    exclude_genres: list[str] = Field(default_factory=lambda: ["Hentai", "Erotica"])
    exclude_rating_prefixes: list[str] = Field(default_factory=lambda: ["Rx"])


class LLMSection(_Section):
    provider: Literal["none", "openai_compatible", "anthropic"] = "none"
    base_url: str = ""
    model: str = ""
    timeout_s: float = 30
    max_output_tokens: int = 700
    temperature: float = 0.2
    max_calls_per_request: int = 2
    effort: str = "low"
    anthropic_max_tokens: int = 16000
    price_per_mtok_input: float | None = None
    price_per_mtok_output: float | None = None
    price_as_of: str | None = None


class Settings(BaseModel):
    """Validated settings. Algorithm sections stay as dicts (validated where used)."""

    model_config = ConfigDict(extra="allow")

    app: AppSection = Field(default_factory=AppSection)
    dataset: DatasetSection = Field(default_factory=DatasetSection)
    paths: PathsSection = Field(default_factory=PathsSection)
    content_rules: ContentRules = Field(default_factory=ContentRules)
    llm: LLMSection = Field(default_factory=LLMSection)
    recommendation: dict[str, Any] = Field(default_factory=dict)
    retrieval: dict[str, Any] = Field(default_factory=dict)
    chat: dict[str, Any] = Field(default_factory=dict)
    planner: dict[str, Any] = Field(default_factory=dict)
    images: dict[str, Any] = Field(default_factory=dict)
    jikan: dict[str, Any] = Field(default_factory=dict)
    pipeline: dict[str, Any] = Field(default_factory=dict)
    synthetic: dict[str, Any] = Field(default_factory=dict)
    real: dict[str, Any] = Field(default_factory=dict)
    # secrets / per-process overrides captured from the environment at load time (never dumped)
    _llm_api_key: str = PrivateAttr(default="")
    _db_path_override: str | None = PrivateAttr(default=None)

    # ------------------------------------------------------------------ paths
    @property
    def mode(self) -> str:
        return self.dataset.mode

    def _abs(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def data_root(self) -> Path:
        return self._abs(self.paths.data_root)

    @property
    def artifacts_dir(self) -> Path:
        return self._abs(self.paths.artifacts_root) / self.mode

    @property
    def var_dir(self) -> Path:
        return self._abs(self.paths.var_root) / self.mode

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed" / self.mode

    @property
    def db_path(self) -> Path:
        return Path(self._db_path_override) if self._db_path_override else self.var_dir / "anilogic.sqlite3"

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.db_path}"

    @property
    def evaluation_runs_dir(self) -> Path:
        return self._abs(self.paths.evaluation_runs)

    @property
    def synthetic_dir(self) -> Path:
        seed = self.synthetic.get("seed", 42)
        return self.data_root / "synthetic" / f"seed{seed}"

    @property
    def llm_api_key(self) -> str:
        return self._llm_api_key

    def section(self, *keys: str) -> dict[str, Any]:
        node: Any = self.model_dump()
        for k in keys:
            node = node.get(k, {}) if isinstance(node, dict) else {}
        return node if isinstance(node, dict) else {}

    def fingerprint(self, *keys: str) -> str:
        """Stable hash of a config sub-tree (used in manifests / cache keys)."""
        payload = json.dumps(self.section(*keys) if keys else self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _coerce(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _env_overrides(environ: dict[str, str]) -> dict:
    out: dict = {}
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX) or "__" not in key:
            continue
        path = key[len(ENV_PREFIX):].lower().split("__")
        node = out
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = _coerce(raw)
    return out


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def read_dotenv(path: Path) -> dict[str, str]:
    """ANILOGIC_* assignments from a .env file (KEY=value, optional quotes, # comments, optional `export`)."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key.startswith(ENV_PREFIX) and value:
            out[key] = value
    return out


def load_settings(mode: str | None = None, environ: dict[str, str] | None = None,
                  extra_files: list[Path] | None = None) -> Settings:
    if environ is None:
        # the project's .env fills in what the process environment does not set (real variables win)
        environ = {**read_dotenv(PROJECT_ROOT / ".env"), **os.environ}
    environ = dict(environ)
    mode = mode or environ.get(ENV_PREFIX + "DATASET_MODE") or _read_yaml(CONFIG_DIR / "base.yaml").get(
        "dataset", {}).get("mode", "synthetic")
    if mode not in ("synthetic", "real"):
        raise ValueError(f"unknown dataset mode {mode!r}")
    merged = _read_yaml(CONFIG_DIR / "base.yaml")
    merged = _deep_merge(merged, _read_yaml(CONFIG_DIR / f"{mode}.yaml"))
    for extra in extra_files or []:
        merged = _deep_merge(merged, _read_yaml(Path(extra)))
    config_file = environ.get(ENV_PREFIX + "CONFIG")
    if config_file:
        merged = _deep_merge(merged, _read_yaml(Path(config_file)))
    merged = _deep_merge(merged, _env_overrides(environ))
    merged.setdefault("dataset", {})["mode"] = mode
    settings = Settings.model_validate(merged)
    settings._llm_api_key = environ.get(ENV_PREFIX + "LLM_API_KEY", "")
    settings._db_path_override = environ.get(ENV_PREFIX + "DB_PATH") or None
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
