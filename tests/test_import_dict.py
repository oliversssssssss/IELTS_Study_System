"""Tests for ECDICT vocabulary imports."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from src.ielts.vocab.import_dict import app, import_ecdict

SCHEMA_PATH = Path(__file__).parent.parent / "src" / "ielts" / "schema" / "01_vocab.sql"
CSV_HEADER = (
    "word,phonetic,definition,translation,pos,collins,oxford,tag,bnc,frq,"
    "exchange,detail,audio\n"
)


def _execute_schema(db_path: Path) -> None:
    """Execute the vocabulary schema against a SQLite database file."""
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(schema_sql)


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection configured for import assertions."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _write_csv(csv_path: Path, rows: list[str]) -> None:
    """Write a small ECDICT-like CSV fixture."""
    csv_path.write_text(CSV_HEADER + "".join(rows), encoding="utf-8")


def _count_rows(db_path: Path, table_name: str) -> int:
    """Count rows in one vocabulary table."""
    with _connect(db_path) as conn:
        row = conn.execute(f"SELECT COUNT(*) AS row_count FROM {table_name}").fetchone()
    return int(row["row_count"])


def test_import_ecdict_imports_exam_words_and_mastery(tmp_path: Path) -> None:
    """Tagged exam words should be inserted with unknown mastery rows."""
    db_path = tmp_path / "ielts.sqlite"
    csv_path = tmp_path / "ecdict.csv"
    _execute_schema(db_path)
    _write_csv(
        csv_path,
        [
            (
                '"Ambitious",æmˈbɪʃəs,full of ambition,'
                '"adj. 有雄心的; 有野心的",adj,,,ielts gre,123,45,,,\n'
            ),
            (
                '"environment",ɪnˈvaɪrənmənt,'
                '"the surroundings in which a person lives",'
                '"n. 环境\\nn. 自然环境",n,,,toefl ielts,234,56,,,\n'
            ),
            '"study",ˈstʌdi,learn about a subject,"v. 学习",v,,,gk zk,1,2,,,\n',
        ],
    )

    stats = import_ecdict(csv_path=csv_path, db_path=db_path)

    assert stats.imported == 2
    assert stats.skipped == 1
    assert stats.failed == 0
    assert stats.tag_distribution == {"ielts": 2, "gre": 1, "toefl": 1}
    assert _count_rows(db_path, "words") == 2
    assert _count_rows(db_path, "mastery") == 2

    with _connect(db_path) as conn:
        word = conn.execute(
            """
            SELECT lemma, ipa, pos, definitions, tags, bnc_freq, coca_freq
            FROM words
            WHERE lemma = ?
            """,
            ("ambitious",),
        ).fetchone()
        mastery = conn.execute(
            """
            SELECT status
            FROM mastery
            JOIN words ON words.id = mastery.word_id
            WHERE words.lemma = ?
            """,
            ("ambitious",),
        ).fetchone()

    assert word["lemma"] == "ambitious"
    assert word["ipa"] == "æmˈbɪʃəs"
    assert word["pos"] == "adj"
    assert json.loads(word["tags"]) == ["ielts", "gre"]
    assert json.loads(word["definitions"]) == [
        {
            "sense_zh": "adj. 有雄心的",
            "sense_en": "full of ambition",
            "pos": "adj",
        },
        {
            "sense_zh": "有野心的",
            "sense_en": "",
            "pos": "adj",
        },
    ]
    assert word["bnc_freq"] == 123
    assert word["coca_freq"] == 45
    assert mastery["status"] == "unknown"


def test_import_ecdict_counts_skipped_and_failed_rows(tmp_path: Path) -> None:
    """Invalid words are skipped and malformed rows are counted as failures."""
    db_path = tmp_path / "ielts.sqlite"
    csv_path = tmp_path / "ecdict.csv"
    _execute_schema(db_path)
    _write_csv(
        csv_path,
        [
            '"valid",ˈvælɪd,correct,"adj. 有效的",adj,,,ielts,1,2,,,\n',
            '"has space",,definition,"translation",n,,,ielts,1,2,,,\n',
            '"has-hyphen",,definition,"translation",n,,,ielts,1,2,,,\n',
            '"a",,definition,"translation",n,,,ielts,1,2,,,\n',
            '"emptytag",,definition,"translation",n,,,,1,2,,,\n',
            '"general",,definition,"translation",n,,,gk zk,1,2,,,\n',
            '"badint",,definition,"translation",n,,,ielts,not-int,2,,,\n',
            '"malformed",only,twelve,fields\n',
        ],
    )

    stats = import_ecdict(csv_path=csv_path, db_path=db_path)

    assert stats.imported == 1
    assert stats.skipped == 5
    assert stats.failed == 2
    assert _count_rows(db_path, "words") == 1
    assert _count_rows(db_path, "mastery") == 1


def test_import_ecdict_is_idempotent_and_preserves_existing_mastery(
    tmp_path: Path,
) -> None:
    """A repeated import should skip existing lemmas and not alter their mastery."""
    db_path = tmp_path / "ielts.sqlite"
    csv_path = tmp_path / "ecdict.csv"
    _execute_schema(db_path)
    _write_csv(
        csv_path,
        ['"valid",ˈvælɪd,correct,"adj. 有效的",adj,,,ielts,1,2,,,\n'],
    )

    first_stats = import_ecdict(csv_path=csv_path, db_path=db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE mastery
            SET status = 'learning'
            WHERE word_id = (SELECT id FROM words WHERE lemma = ?)
            """,
            ("valid",),
        )

    second_stats = import_ecdict(csv_path=csv_path, db_path=db_path)

    assert first_stats.imported == 1
    assert second_stats.imported == 0
    assert second_stats.skipped == 1
    assert _count_rows(db_path, "words") == 1
    assert _count_rows(db_path, "mastery") == 1
    with _connect(db_path) as conn:
        status = conn.execute("SELECT status FROM mastery").fetchone()["status"]
    assert status == "learning"


def test_import_ecdict_accepts_custom_tag_filter(tmp_path: Path) -> None:
    """A caller can override the default exam tag filter."""
    db_path = tmp_path / "ielts.sqlite"
    csv_path = tmp_path / "ecdict.csv"
    _execute_schema(db_path)
    _write_csv(
        csv_path,
        [
            '"exam",,definition,"translation",n,,,ielts,1,2,,,\n',
            '"school",,definition,"translation",n,,,gk zk,1,2,,,\n',
        ],
    )

    stats = import_ecdict(csv_path=csv_path, db_path=db_path, tag_filter=["zk"])

    assert stats.imported == 1
    assert stats.skipped == 1
    with _connect(db_path) as conn:
        lemma = conn.execute("SELECT lemma FROM words").fetchone()["lemma"]
    assert lemma == "school"


def test_cli_import_tags_and_reset(tmp_path: Path) -> None:
    """CLI options should import custom tags and reset existing rows after confirmation."""
    runner = CliRunner()
    db_path = tmp_path / "ielts.sqlite"
    csv_path = tmp_path / "ecdict.csv"
    _execute_schema(db_path)
    _write_csv(
        csv_path,
        [
            '"exam",,definition,"translation",n,,,ielts,1,2,,,\n',
            '"school",,definition,"translation",n,,,zk,1,2,,,\n',
        ],
    )

    first_result = runner.invoke(
        app,
        [
            "--csv-path",
            str(csv_path),
            "--db-path",
            str(db_path),
            "--tags",
            "zk",
        ],
    )
    assert first_result.exit_code == 0
    assert "Imported 1 words from ECDICT" in first_result.output
    assert _count_rows(db_path, "words") == 1

    reset_result = runner.invoke(
        app,
        [
            "--csv-path",
            str(csv_path),
            "--db-path",
            str(db_path),
            "--tags",
            "ielts",
            "--reset",
        ],
        input="y\n",
    )
    assert reset_result.exit_code == 0
    assert "Imported 1 words from ECDICT" in reset_result.output
    assert _count_rows(db_path, "words") == 1
    with _connect(db_path) as conn:
        lemma = conn.execute("SELECT lemma FROM words").fetchone()["lemma"]
    assert lemma == "exam"
