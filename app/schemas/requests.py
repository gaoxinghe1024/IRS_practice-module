"""Request bodies (validated by Pydantic; unknown fields are rejected)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Credentials(Strict):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class TagRef(Strict):
    namespace: Literal["genre", "theme", "demographic", "studio", "source"]
    value: str = Field(min_length=1, max_length=120)


class PreferencesUpdate(Strict):
    liked_tags: list[TagRef] | None = None
    length_preference: Literal["any", "short", "medium", "long"] | None = None
    hide_watched_in_similar: bool | None = None


class RatingBody(Strict):
    rating: int = Field(ge=1, le=10)


class ProgressBody(Strict):
    status: Literal["plan_to_watch", "watching", "completed", "on_hold", "dropped"]
    watched_episodes: int | None = Field(default=None, ge=0, le=100000)


class CollectionCreate(Strict):
    name: str = Field(min_length=1, max_length=60)
    description: str = Field(default="", max_length=300)


class CollectionUpdate(Strict):
    name: str | None = Field(default=None, min_length=1, max_length=60)
    description: str | None = Field(default=None, max_length=300)


class Relaxation(Strict):
    rule_id: str
    change: dict


class CollectionRecRequest(Strict):
    keywords: str = Field(default="", max_length=200)
    include_tags: list[TagRef] = Field(default_factory=list, max_length=10)
    min_score: float | None = Field(default=None, ge=0, le=10)
    max_episodes: int | None = Field(default=None, ge=1, le=10000)
    types: list[str] | None = Field(default=None, max_length=8)
    k: int | None = Field(default=None, ge=1, le=20)
    scorer: str | None = None
    relaxation_applied: Relaxation | None = None


class HomeRequest(Strict):
    k: int | None = Field(default=None, ge=1, le=30)
    engine: Literal["icf", "mf", "ncf"] | None = None
    reranker: Literal["ga", "greedy", "none"] | None = None
    candidate_k: int | None = Field(default=None, ge=1, le=200)


class FeedbackBody(Strict):
    event_type: str = Field(max_length=32)
    item_key: str | None = Field(default=None, max_length=40)
    request_id: str | None = Field(default=None, max_length=32)
    surface: str | None = Field(default=None, max_length=32)
    algorithm: str | None = Field(default=None, max_length=64)
    value: str | None = Field(default=None, max_length=64)
    context: dict = Field(default_factory=dict)


class ExclusionBody(Strict):
    kind: Literal["item", "tag"]
    item_key: str | None = Field(default=None, max_length=40)
    tag: TagRef | None = None


class ChatSessionCreate(Strict):
    title: str | None = Field(default=None, max_length=120)


class ChatMessageBody(Strict):
    message: str = Field(min_length=1, max_length=1000)
    save_preferences: bool = False


class DailySlot(Strict):
    label: str = Field(min_length=1, max_length=60)
    minutes: int = Field(ge=1, le=24 * 60)


class PlanRequest(Strict):
    budget_minutes: int = Field(ge=10, le=60 * 24 * 14)
    max_titles: int = Field(default=5, ge=1, le=20)
    mode: Literal["complete", "partial"] = "complete"
    include_genres: list[str] = Field(default_factory=list, max_length=10)
    exclude_genres: list[str] = Field(default_factory=list, max_length=10)
    min_score: float | None = Field(default=None, ge=0, le=10)
    max_episodes_per_title: int | None = Field(default=None, ge=1, le=5000)
    diversity: Literal["low", "normal", "high"] = "normal"
    allow_prerequisites: bool = True
    break_minutes_per_title: int = Field(default=0, ge=0, le=120)
    daily_slots: list[DailySlot] = Field(default_factory=list, max_length=31)
    must_include: list[str] = Field(default_factory=list, max_length=10)
    title: str | None = Field(default=None, max_length=120)


class PlanItemEdit(Strict):
    item_key: str
    episode_start: int | None = Field(default=None, ge=1)
    episode_end: int | None = Field(default=None, ge=1)


class PlanUpdate(Strict):
    title: str | None = Field(default=None, max_length=120)
    remove_items: list[str] = Field(default_factory=list)
    edit_items: list[PlanItemEdit] = Field(default_factory=list)
    order: list[str] | None = None
    request: PlanRequest | None = None


class SceneSearchBody(Strict):
    query: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=10)
