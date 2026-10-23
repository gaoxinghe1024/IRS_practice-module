"""serving data in the database: full catalogue order, training counts, supplement tags, embedding vectors

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-19
"""
from alembic import op
import sqlalchemy as sa


revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Plain ALTER TABLE, never a batch (copy-and-drop) rebuild: anime is the parent of user data with ON DELETE
    # CASCADE foreign keys.
    op.add_column("anime", sa.Column("exclusion_reasons", sa.JSON(), nullable=True))
    op.add_column("anime", sa.Column("catalog_position", sa.Integer(), nullable=True))
    op.add_column("anime", sa.Column("train_interactions", sa.Integer(), nullable=True))
    op.create_index("ix_anime_catalog_position", "anime", ["catalog_position"], unique=True)

    op.create_table("supplement_tags",
    sa.Column("mal_id", sa.Integer(), nullable=False),
    sa.Column("namespace", sa.String(length=16), nullable=False),
    sa.Column("position", sa.Integer(), nullable=False),
    sa.Column("value", sa.String(length=120), nullable=False),
    sa.PrimaryKeyConstraint("mal_id", "namespace", "position")
    )
    op.create_table("vector_sets",
    sa.Column("name", sa.String(length=32), nullable=False),
    sa.Column("dataset_version", sa.String(length=64), nullable=False),
    sa.Column("dim", sa.Integer(), nullable=False),
    sa.Column("dtype", sa.String(length=16), nullable=False),
    sa.Column("count", sa.Integer(), nullable=False),
    sa.Column("sha256", sa.String(length=64), nullable=False),
    sa.Column("info", sa.JSON(), nullable=False),
    sa.PrimaryKeyConstraint("name")
    )
    op.create_table("anime_embeddings",
    sa.Column("item_key", sa.String(length=40), nullable=False),
    sa.Column("vector", sa.LargeBinary(), nullable=False),
    sa.Column("has_vector", sa.Boolean(), nullable=False),
    sa.ForeignKeyConstraint(["item_key"], ["anime.item_key"], ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("item_key")
    )
    op.create_table("document_embeddings",
    sa.Column("doc_id", sa.String(length=80), nullable=False),
    sa.Column("position", sa.Integer(), nullable=False),
    sa.Column("vector", sa.LargeBinary(), nullable=False),
    sa.ForeignKeyConstraint(["doc_id"], ["knowledge_documents.doc_id"], ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("doc_id"),
    sa.UniqueConstraint("position")
    )
    op.create_table("scene_image_embeddings",
    sa.Column("image_id", sa.String(length=64), nullable=False),
    sa.Column("position", sa.Integer(), nullable=False),
    sa.Column("vector", sa.LargeBinary(), nullable=False),
    sa.ForeignKeyConstraint(["image_id"], ["scene_images.image_id"], ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("image_id"),
    sa.UniqueConstraint("position")
    )


def downgrade() -> None:
    op.drop_table("scene_image_embeddings")
    op.drop_table("document_embeddings")
    op.drop_table("anime_embeddings")
    op.drop_table("vector_sets")
    op.drop_table("supplement_tags")
    op.drop_index("ix_anime_catalog_position", table_name="anime")
    for col in ("train_interactions", "catalog_position", "exclusion_reasons"):
        op.execute(f"ALTER TABLE anime DROP COLUMN {col}")   # SQLite >= 3.35; no table rebuild
