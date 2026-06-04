"""Tests for database initialization."""

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from src.ielts.schema import init_db as init_db_module
from src.ielts.schema.init_db import init_db


EXPECTED_TABLES = [
    "collocations",
    "mastery",
    "word_occurrences",
    "words",
]


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection configured for init-db assertions."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _table_names(db_path: Path) -> list[str]:
    """Return sorted user table names from a SQLite database."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
    return [row["name"] for row in rows]


def test_init_db_creates_new_database(tmp_path: Path) -> None:
    """A fresh database path should be created and initialized."""
    db_path = tmp_path / "nested" / "ielts.sqlite"

    init_db(db_path)

    assert db_path.exists()
    assert _table_names(db_path) == EXPECTED_TABLES


def test_init_db_is_idempotent(tmp_path: Path) -> None:
    """Running initialization twice against the same file should not fail."""
    db_path = tmp_path / "ielts.sqlite"

    init_db(db_path)
    init_db(db_path)

    assert _table_names(db_path) == EXPECTED_TABLES


def test_init_db_creates_all_required_tables(tmp_path: Path) -> None:
    """Initialization should create the four vocabulary tables."""
    db_path = tmp_path / "ielts.sqlite"

    init_db(db_path)

    assert _table_names(db_path) == EXPECTED_TABLES


def test_init_db_enables_wal_mode(tmp_path: Path) -> None:
    """Initialization should persist WAL journaling mode for the database."""
    db_path = tmp_path / "ielts.sqlite"
    init_db(db_path)

    with _connect(db_path) as conn:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

    assert journal_mode == "wal"


def test_init_db_enables_foreign_keys_on_setup_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initialization should enable foreign-key checks on its SQLite connection."""
    db_path = tmp_path / "ielts.sqlite"
    statements: list[str] = []
    original_connect = sqlite3.connect

    class RecordingConnection(sqlite3.Connection):
        """SQLite connection that records executed SQL statements."""

        def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
            statements.append(sql)
            return super().execute(sql, parameters)

    def recording_connect(database: str | Path, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        kwargs["factory"] = RecordingConnection
        return original_connect(database, *args, **kwargs)

    monkeypatch.setattr(init_db_module.sqlite3, "connect", recording_connect)

    init_db(db_path)

    normalized_statements = [statement.upper().replace(" ", "") for statement in statements]
    assert "PRAGMAFOREIGN_KEYS=ON" in normalized_statements


def test_init_db_missing_schema_directory_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing schema directory should raise a clear FileNotFoundError."""
    missing_schema_dir = tmp_path / "missing-schema"
    db_path = tmp_path / "ielts.sqlite"
    monkeypatch.setattr(init_db_module, "SCHEMA_DIR", missing_schema_dir)

    with pytest.raises(FileNotFoundError, match="Schema directory does not exist"):
        init_db(db_path)
