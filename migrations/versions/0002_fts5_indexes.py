"""FTS5 full-text indexes for catalogue titles/aliases and knowledge documents

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-30
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Contentless-style standalone FTS tables, rebuilt on dataset activation.
    op.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS anime_fts USING fts5("
        "item_key UNINDEXED, title, aliases, tokenize='unicode61 remove_diacritics 2')"
    )
    op.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5("
        "doc_id UNINDEXED, item_key UNINDEXED, kind UNINDEXED, title, body, "
        "tokenize='porter unicode61 remove_diacritics 2')"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS docs_fts")
    op.execute("DROP TABLE IF EXISTS anime_fts")
