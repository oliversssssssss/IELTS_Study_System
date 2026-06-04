"""Tests for the IELTS vocabulary database schema."""

import sqlite3
from pathlib import Path

import pytest

SCHEMA_PATH = Path(__file__).parent.parent / "src" / "ielts" / "schema" / "01_vocab.sql"


def _execute_schema(db_path: Path) -> None:
    """Execute the vocabulary schema against a SQLite database file."""
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(schema_sql)


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection configured for schema assertions."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def test_schema_executes_and_creates_required_tables(tmp_path: Path) -> None:
    """The schema should execute cleanly and create the four requested tables."""
    db_path = tmp_path / "vocab.db"

    _execute_schema(db_path)

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

    assert [row["name"] for row in rows] == [
        "collocations",
        "mastery",
        "word_occurrences",
        "words",
    ]


def test_schema_execution_is_idempotent(tmp_path: Path) -> None:
    """Running the schema twice against the same database should not fail."""
    db_path = tmp_path / "vocab.db"

    _execute_schema(db_path)
    _execute_schema(db_path)

    with _connect(db_path) as conn:
        table_count = conn.execute(
            """
            SELECT COUNT(*) AS table_count
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            """
        ).fetchone()["table_count"]

    assert table_count == 4


def test_mastery_status_check_constraint_rejects_invalid_status(tmp_path: Path) -> None:
    """The mastery status state machine should reject unknown states."""
    db_path = tmp_path / "vocab.db"
    _execute_schema(db_path)

    with _connect(db_path) as conn:
        conn.execute("INSERT INTO words (lemma) VALUES (?)", ("test",))

        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            conn.execute(
                "INSERT INTO mastery (word_id, status) VALUES (?, ?)",
                (1, "invalid_status"),
            )


def test_word_occurrences_source_type_check_constraint_rejects_invalid_source(
    tmp_path: Path,
) -> None:
    """The occurrence source type should reject unsupported source names."""
    db_path = tmp_path / "vocab.db"
    _execute_schema(db_path)

    with _connect(db_path) as conn:
        conn.execute("INSERT INTO words (lemma) VALUES (?)", ("test",))

        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            conn.execute(
                "INSERT INTO word_occurrences (word_id, source_type) VALUES (?, ?)",
                (1, "invalid_source"),
            )


def test_words_lemma_unique_constraint_is_case_insensitive(tmp_path: Path) -> None:
    """The lemma unique constraint should treat case variants as duplicates."""
    db_path = tmp_path / "vocab.db"
    _execute_schema(db_path)

    with _connect(db_path) as conn:
        conn.execute("INSERT INTO words (lemma) VALUES (?)", ("test",))

        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
            conn.execute("INSERT INTO words (lemma) VALUES (?)", ("Test",))


def test_mastery_is_one_to_one_with_words(tmp_path: Path) -> None:
    """A word should have at most one mastery row."""
    db_path = tmp_path / "vocab.db"
    _execute_schema(db_path)

    with _connect(db_path) as conn:
        conn.execute("INSERT INTO words (lemma) VALUES (?)", ("test",))
        conn.execute("INSERT INTO mastery (word_id, status) VALUES (?, ?)", (1, "unknown"))

        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
            conn.execute("INSERT INTO mastery (word_id, status) VALUES (?, ?)", (1, "learning"))


def test_mastery_does_not_cascade_delete_with_words(tmp_path: Path) -> None:
    """Deleting a referenced word should not cascade-delete mastery rows."""
    db_path = tmp_path / "vocab.db"
    _execute_schema(db_path)

    with _connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("INSERT INTO words (lemma) VALUES (?)", ("test",))
        conn.execute("INSERT INTO mastery (word_id, status) VALUES (?, ?)", (1, "unknown"))

        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
            conn.execute("DELETE FROM words WHERE id = ?", (1,))

        mastery_count = conn.execute(
            "SELECT COUNT(*) AS mastery_count FROM mastery WHERE word_id = ?",
            (1,),
        ).fetchone()["mastery_count"]

    assert mastery_count == 1
