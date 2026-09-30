"""Business database schema (SQLite via SQLAlchemy 2.x).

Offline training data (the 24M Kaggle ratings, splits, matrices) is NOT stored
here; it lives in data/processed and artifacts/. This database holds the served
catalogue for the active dataset version plus everything application users do.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


# --------------------------------------------------------------------------- users


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    normalized_username: Mapped[str] = mapped_column(String(40), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(40), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    profile_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    onboarding_state: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)
    origin: Mapped[str] = mapped_column(String(32), default="registered", nullable=False)


class UserSession(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    csrf_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


# --------------------------------------------------------------------------- catalogue


class Anime(Base):
    __tablename__ = "anime"

    item_key: Mapped[str] = mapped_column(String(40), primary_key=True)
    mal_id: Mapped[int | None] = mapped_column(Integer, unique=True, nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    title_english: Mapped[str | None] = mapped_column(Text)
    title_other: Mapped[str | None] = mapped_column(Text)
    synopsis: Mapped[str | None] = mapped_column(Text)
    score: Mapped[float | None] = mapped_column(Float)          # catalogue aggregate (MAL 2023 snapshot)
    scored_by: Mapped[int | None] = mapped_column(Integer)
    rank: Mapped[int | None] = mapped_column(Integer)
    popularity: Mapped[int | None] = mapped_column(Integer)
    members: Mapped[int | None] = mapped_column(Integer)
    favorites: Mapped[int | None] = mapped_column(Integer)
    type: Mapped[str | None] = mapped_column(String(16))
    episodes: Mapped[int | None] = mapped_column(Integer)
    duration_minutes: Mapped[float | None] = mapped_column(Float)  # per episode
    total_minutes: Mapped[float | None] = mapped_column(Float)
    duration_raw: Mapped[str | None] = mapped_column(String(64))
    duration_basis: Mapped[str | None] = mapped_column(String(32))  # per_episode_average | total_runtime
    status: Mapped[str | None] = mapped_column(String(32))
    content_rating: Mapped[str | None] = mapped_column(String(64))
    source_material: Mapped[str | None] = mapped_column(String(64))
    aired_raw: Mapped[str | None] = mapped_column(String(80))
    aired_from: Mapped[str | None] = mapped_column(String(10))
    aired_to: Mapped[str | None] = mapped_column(String(10))
    premiered_season: Mapped[str | None] = mapped_column(String(8))
    premiered_year: Mapped[int | None] = mapped_column(Integer)
    cover_url: Mapped[str | None] = mapped_column(Text)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    missing_fields: Mapped[list[Any]] = mapped_column(JSON, default=list)
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False)
    in_catalog: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_recommendable: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_planner_eligible: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_explicit: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    tags: Mapped[list[AnimeTag]] = relationship(back_populates="anime", cascade="all, delete-orphan")


class AnimeAlias(Base):
    __tablename__ = "anime_aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_key: Mapped[str] = mapped_column(ForeignKey("anime.item_key", ondelete="CASCADE"), index=True)
    alias: Mapped[str] = mapped_column(Text, nullable=False)
    normalized: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)


class Tag(Base):
    __tablename__ = "tags"
    __table_args__ = (UniqueConstraint("namespace", "normalized"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    namespace: Mapped[str] = mapped_column(String(16), nullable=False)  # genre|theme|demographic|studio|producer|licensor|source
    value: Mapped[str] = mapped_column(String(120), nullable=False)
    normalized: Mapped[str] = mapped_column(String(120), nullable=False)


class AnimeTag(Base):
    __tablename__ = "anime_tags"

    item_key: Mapped[str] = mapped_column(ForeignKey("anime.item_key", ondelete="CASCADE"), primary_key=True)
    tag_id: Mapped[int] = mapped_column(ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True, index=True)
    source: Mapped[str] = mapped_column(String(24), nullable=False)

    anime: Mapped[Anime] = relationship(back_populates="tags")
    tag: Mapped[Tag] = relationship()


class Relation(Base):
    __tablename__ = "relations"
    __table_args__ = (Index("ix_relations_from_type", "from_item", "relation_type"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    from_item: Mapped[str] = mapped_column(ForeignKey("anime.item_key", ondelete="CASCADE"), nullable=False)
    to_item: Mapped[str | None] = mapped_column(ForeignKey("anime.item_key", ondelete="CASCADE"), nullable=True)
    to_entity_type: Mapped[str] = mapped_column(String(16), default="anime", nullable=False)
    to_external_id: Mapped[str | None] = mapped_column(String(40))
    to_name: Mapped[str | None] = mapped_column(Text)
    relation_type: Mapped[str] = mapped_column(String(32), nullable=False)  # prequel|sequel|side_story|...
    status: Mapped[str] = mapped_column(String(16), default="confirmed", nullable=False)  # confirmed|inferred
    source_ref: Mapped[str] = mapped_column(Text, nullable=False)


class Character(Base):
    __tablename__ = "characters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[str] = mapped_column(String(40), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class AnimeCharacter(Base):
    __tablename__ = "anime_characters"

    item_key: Mapped[str] = mapped_column(ForeignKey("anime.item_key", ondelete="CASCADE"), primary_key=True)
    character_id: Mapped[int] = mapped_column(ForeignKey("characters.id", ondelete="CASCADE"), primary_key=True)
    role: Mapped[str | None] = mapped_column(String(24))
    source_ref: Mapped[str] = mapped_column(Text, nullable=False)


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    doc_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    item_key: Mapped[str] = mapped_column(ForeignKey("anime.item_key", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    spoiler_level: Mapped[str] = mapped_column(String(12), default="none", nullable=False)
    language: Mapped[str] = mapped_column(String(8), default="en", nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_ref: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    locator: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # parent / chunk / char span
    retrieved_at: Mapped[str | None] = mapped_column(String(32))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False)


class Scene(Base):
    __tablename__ = "scenes"

    scene_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    item_key: Mapped[str] = mapped_column(ForeignKey("anime.item_key", ondelete="CASCADE"), index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    episode: Mapped[int | None] = mapped_column(Integer)
    timestamp_seconds: Mapped[int | None] = mapped_column(Integer)
    characters: Mapped[list[Any]] = mapped_column(JSON, default=list)
    source_ref: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    spoiler_level: Mapped[str] = mapped_column(String(12), default="major")
    verification_status: Mapped[str] = mapped_column(String(16), default="unverified")


class SceneImage(Base):
    __tablename__ = "scene_images"

    image_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    scene_id: Mapped[str | None] = mapped_column(ForeignKey("scenes.scene_id", ondelete="SET NULL"))
    item_key: Mapped[str] = mapped_column(ForeignKey("anime.item_key", ondelete="CASCADE"), index=True)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    episode: Mapped[int | None] = mapped_column(Integer)
    timestamp_seconds: Mapped[int | None] = mapped_column(Integer)
    split: Mapped[str] = mapped_column(String(12), default="gallery")
    provenance: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


# --------------------------------------------------------------------------- user data


class Rating(Base):
    __tablename__ = "ratings"
    __table_args__ = (
        UniqueConstraint("user_id", "item_key"),
        CheckConstraint("rating >= 1 AND rating <= 10", name="ck_rating_range"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    item_key: Mapped[str] = mapped_column(ForeignKey("anime.item_key"), nullable=False, index=True)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class Collection(Base):
    __tablename__ = "collections"
    __table_args__ = (UniqueConstraint("user_id", "name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(60), nullable=False)
    description: Mapped[str] = mapped_column(String(300), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    items: Mapped[list[CollectionItem]] = relationship(back_populates="collection", cascade="all, delete-orphan")


class CollectionItem(Base):
    __tablename__ = "collection_items"

    collection_id: Mapped[int] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"), primary_key=True)
    item_key: Mapped[str] = mapped_column(ForeignKey("anime.item_key"), primary_key=True, index=True)
    added_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    collection: Mapped[Collection] = relationship(back_populates="items")


class WatchProgress(Base):
    __tablename__ = "watch_progress"
    __table_args__ = (
        CheckConstraint("watched_episodes >= 0", name="ck_progress_nonneg"),
        CheckConstraint(
            "status IN ('plan_to_watch','watching','completed','on_hold','dropped')", name="ck_progress_status"),
    )

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    item_key: Mapped[str] = mapped_column(ForeignKey("anime.item_key"), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    watched_episodes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class Preference(Base):
    """Account-level preferences (explicitly saved). Chat-session constraints live in chat state."""

    __tablename__ = "preferences"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    liked_tag_ids: Mapped[list[Any]] = mapped_column(JSON, default=list)
    length_preference: Mapped[str] = mapped_column(String(12), default="any", nullable=False)
    hide_watched_in_similar: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class Exclusion(Base):
    __tablename__ = "exclusions"
    __table_args__ = (
        CheckConstraint("(kind = 'item' AND item_key IS NOT NULL AND tag_id IS NULL) OR "
                        "(kind = 'tag' AND tag_id IS NOT NULL AND item_key IS NULL)", name="ck_exclusion_kind"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(8), nullable=False)
    item_key: Mapped[str | None] = mapped_column(ForeignKey("anime.item_key"))
    tag_id: Mapped[int | None] = mapped_column(ForeignKey("tags.id"))
    source: Mapped[str] = mapped_column(String(32), default="not_interested", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime)


class FeedbackEvent(Base):
    __tablename__ = "feedback_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    request_id: Mapped[str | None] = mapped_column(String(32))
    item_key: Mapped[str | None] = mapped_column(ForeignKey("anime.item_key"))
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    value: Mapped[str | None] = mapped_column(String(64))
    is_explicit: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    surface: Mapped[str | None] = mapped_column(String(32))
    algorithm: Mapped[str | None] = mapped_column(String(64))
    context: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


# --------------------------------------------------------------------------- chat / plans / runs


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(120), default="New chat", nullable=False)
    state: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    messages: Mapped[list[ChatMessage]] = relationship(back_populates="session", cascade="all, delete-orphan",
                                                       order_by="ChatMessage.id")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(12), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    parsed: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    model_info: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    dataset_version: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class ViewingPlan(Base):
    __tablename__ = "viewing_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(120), nullable=False)
    current_revision: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    revisions: Mapped[list[PlanRevision]] = relationship(back_populates="plan", cascade="all, delete-orphan",
                                                         order_by="PlanRevision.revision")


class PlanRevision(Base):
    __tablename__ = "plan_revisions"
    __table_args__ = (UniqueConstraint("plan_id", "revision"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    plan_id: Mapped[int] = mapped_column(ForeignKey("viewing_plans.id", ondelete="CASCADE"), index=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    request: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    feasibility: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    algorithm: Mapped[str] = mapped_column(String(48), nullable=False)
    algorithm_version: Mapped[str] = mapped_column(String(32), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False)
    change_note: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    plan: Mapped[ViewingPlan] = relationship(back_populates="revisions")
    items: Mapped[list[PlanItem]] = relationship(back_populates="revision", cascade="all, delete-orphan",
                                                 order_by="PlanItem.order_index")
    sessions: Mapped[list[PlanSession]] = relationship(back_populates="revision", cascade="all, delete-orphan",
                                                       order_by="PlanSession.session_index")


class PlanItem(Base):
    __tablename__ = "plan_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    revision_id: Mapped[int] = mapped_column(ForeignKey("plan_revisions.id", ondelete="CASCADE"), index=True)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)
    item_key: Mapped[str] = mapped_column(ForeignKey("anime.item_key"), nullable=False)
    episode_start: Mapped[int] = mapped_column(Integer, nullable=False)
    episode_end: Mapped[int] = mapped_column(Integer, nullable=False)
    minutes: Mapped[float] = mapped_column(Float, nullable=False)
    is_full: Mapped[bool] = mapped_column(Boolean, nullable=False)
    reasons: Mapped[list[Any]] = mapped_column(JSON, default=list)
    prerequisite_keys: Mapped[list[Any]] = mapped_column(JSON, default=list)
    duration_evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    revision: Mapped[PlanRevision] = relationship(back_populates="items")


class PlanSession(Base):
    __tablename__ = "plan_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    revision_id: Mapped[int] = mapped_column(ForeignKey("plan_revisions.id", ondelete="CASCADE"), index=True)
    session_index: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str] = mapped_column(String(60), nullable=False)
    capacity_minutes: Mapped[float] = mapped_column(Float, nullable=False)
    entries: Mapped[list[Any]] = mapped_column(JSON, default=list)

    revision: Mapped[PlanRevision] = relationship(back_populates="sessions")


class RecommendationRun(Base):
    __tablename__ = "recommendation_runs"

    request_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    surface: Mapped[str] = mapped_column(String(32), nullable=False)
    params_fingerprint: Mapped[str] = mapped_column(String(32), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(64))
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False)
    requested_k: Mapped[int] = mapped_column(Integer, nullable=False)
    actual_k: Mapped[int] = mapped_column(Integer, nullable=False)
    candidates: Mapped[list[Any]] = mapped_column(JSON, default=list)
    results: Mapped[list[Any]] = mapped_column(JSON, default=list)
    trace: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    command: Mapped[str] = mapped_column(String(48), nullable=False)
    dataset_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(32), nullable=False)
    input_hashes: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    outputs: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    error_summary: Mapped[str | None] = mapped_column(Text)


class ArtifactRecord(Base):
    __tablename__ = "artifact_registry"
    __table_args__ = (UniqueConstraint("dataset_mode", "version", "kind"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class SystemState(Base):
    __tablename__ = "system_state"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
