"""Import exam-relevant ECDICT entries into the vocabulary database."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

DEFAULT_CSV_PATH = Path("data/vocab/ecdict.csv")
DEFAULT_DB_PATH = Path("db/ielts.sqlite")
DEFAULT_TAGS = frozenset({"ielts", "toefl", "gre", "gmat", "cet6", "awl"})
CSV_FIELDS = (
    "word",
    "phonetic",
    "definition",
    "translation",
    "pos",
    "collins",
    "oxford",
    "tag",
    "bnc",
    "frq",
    "exchange",
    "detail",
    "audio",
)
BATCH_SIZE = 1000
WORD_PATTERN = re.compile(r"^[a-z]+$")
TRANSLATION_SPLIT_PATTERN = re.compile(r"\n+|;\s*")

app = typer.Typer(add_completion=False)


@dataclass
class ImportStats:
    """Summary of an ECDICT import run."""

    imported: int
    skipped: int
    failed: int
    tag_distribution: dict[str, int]
    duration_seconds: float


@dataclass(frozen=True)
class WordEntry:
    """Cleaned word data ready for database insertion."""

    lemma: str
    ipa: str | None
    pos: str | None
    tags: tuple[str, ...]
    definitions: str
    bnc_freq: int | None
    coca_freq: int | None


def _configure_csv_field_size_limit() -> None:
    """Allow csv.reader to handle very large definition fields."""
    field_limit = sys.maxsize
    while field_limit > 0:
        try:
            csv.field_size_limit(field_limit)
            return
        except OverflowError:
            field_limit //= 10


def _target_tags(tag_filter: list[str] | None) -> frozenset[str]:
    """Normalize API tag filters."""
    if tag_filter is None:
        return DEFAULT_TAGS

    return frozenset(tag.strip().lower() for tag in tag_filter if tag.strip())


def _parse_cli_tags(tags: str | None) -> list[str] | None:
    """Parse comma-separated CLI tags into an API tag filter."""
    if tags is None:
        return None
    return [tag.strip() for tag in tags.split(",") if tag.strip()]


def _parse_tags(raw_tags: str) -> tuple[str, ...]:
    """Parse ECDICT's space-separated tag field."""
    tags = [tag.strip().lower() for tag in raw_tags.split() if tag.strip()]
    return tuple(dict.fromkeys(tags))


def _valid_lemma(raw_word: str) -> str | None:
    """Return a normalized lemma when the ECDICT word is importable."""
    lemma = raw_word.strip().lower()
    if len(lemma) < 2:
        return None
    if WORD_PATTERN.fullmatch(lemma) is None:
        return None
    return lemma


def _parse_optional_int(raw_value: str, field_name: str) -> int | None:
    """Parse an optional integer field from ECDICT."""
    value = raw_value.strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer: {raw_value!r}") from exc


def _split_translation(translation: str) -> list[str]:
    """Split simple translation multi-sense text into individual senses."""
    return [
        part.strip()
        for part in TRANSLATION_SPLIT_PATTERN.split(translation.strip())
        if part.strip()
    ]


def _split_definition(definition: str) -> list[str]:
    """Split English definitions on line breaks only."""
    return [part.strip() for part in definition.splitlines() if part.strip()]


def _definitions_json(translation: str, definition: str, pos: str | None) -> str:
    """Build the JSON value stored in words.definitions."""
    translations = _split_translation(translation)
    definitions = _split_definition(definition)
    if not translations and not definitions:
        return "[]"

    sense_count = max(len(translations), len(definitions), 1)
    senses = []
    for index in range(sense_count):
        senses.append(
            {
                "sense_zh": translations[index] if index < len(translations) else "",
                "sense_en": definitions[index] if index < len(definitions) else "",
                "pos": pos or "",
            }
        )
    return json.dumps(senses, ensure_ascii=False)


def _parse_row(row: list[str], target_tags: frozenset[str]) -> WordEntry | None:
    """Parse and validate one ECDICT CSV row."""
    if len(row) != len(CSV_FIELDS):
        raise ValueError(f"expected {len(CSV_FIELDS)} CSV fields, got {len(row)}")

    data = dict(zip(CSV_FIELDS, row, strict=True))
    lemma = _valid_lemma(data["word"])
    if lemma is None:
        return None

    tags = _parse_tags(data["tag"])
    if not tags:
        return None
    if target_tags.isdisjoint(tags):
        return None

    pos = data["pos"].strip() or None
    return WordEntry(
        lemma=lemma,
        ipa=data["phonetic"].strip() or None,
        pos=pos,
        tags=tags,
        definitions=_definitions_json(
            translation=data["translation"],
            definition=data["definition"],
            pos=pos,
        ),
        bnc_freq=_parse_optional_int(data["bnc"], "bnc"),
        coca_freq=_parse_optional_int(data["frq"], "frq"),
    )


def _ensure_vocab_tables(conn: sqlite3.Connection) -> None:
    """Fail fast when the target database was not initialized."""
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name IN ('words', 'mastery')
        """
    ).fetchall()
    table_names = {row["name"] for row in rows}
    missing_tables = {"words", "mastery"} - table_names
    if missing_tables:
        missing_text = ", ".join(sorted(missing_tables))
        raise sqlite3.OperationalError(f"missing required tables: {missing_text}")


def _word_exists(conn: sqlite3.Connection, lemma: str) -> bool:
    """Return whether a lemma already exists in words."""
    row = conn.execute("SELECT 1 FROM words WHERE lemma = ?", (lemma,)).fetchone()
    return row is not None


def _insert_entry(
    conn: sqlite3.Connection,
    entry: WordEntry,
    *,
    skip_existing: bool,
) -> bool:
    """Insert one cleaned word and its unknown mastery row."""
    if skip_existing and _word_exists(conn, entry.lemma):
        return False

    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO words
            (lemma, pos, ipa, definitions, tags, bnc_freq, coca_freq)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            entry.lemma,
            entry.pos,
            entry.ipa,
            entry.definitions,
            json.dumps(list(entry.tags), ensure_ascii=False),
            entry.bnc_freq,
            entry.coca_freq,
        ),
    )
    inserted = cursor.rowcount == 1

    row = conn.execute("SELECT id FROM words WHERE lemma = ?", (entry.lemma,)).fetchone()
    if row is None:
        raise sqlite3.DatabaseError(f"word insert did not produce an id: {entry.lemma}")

    if inserted or not skip_existing:
        conn.execute(
            "INSERT OR IGNORE INTO mastery (word_id, status) VALUES (?, 'unknown')",
            (int(row["id"]),),
        )

    return inserted


def _is_header(row: list[str]) -> bool:
    """Return whether a CSV row is the expected ECDICT header."""
    return tuple(cell.strip() for cell in row) == CSV_FIELDS


def _build_progress() -> Progress:
    """Create the Rich progress display used during import."""
    console = Console(stderr=True)
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("{task.completed} rows"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
        disable=not console.is_terminal,
    )


def reset_vocab_tables(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Delete all rows from words and mastery.

    Args:
        db_path: SQLite database path.
    """
    target_db_path = Path(db_path)
    with sqlite3.connect(target_db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        _ensure_vocab_tables(conn)
        conn.execute("DELETE FROM mastery")
        conn.execute("DELETE FROM words")


def import_ecdict(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    tag_filter: list[str] | None = None,
    skip_existing: bool = True,
) -> ImportStats:
    """从 ECDICT CSV 导入词库。

    Args:
        csv_path: ECDICT CSV 路径
        db_path: SQLite 数据库路径
        tag_filter: 只导入含这些标签的词。None 表示默认筛选
        skip_existing: 跳过 lemma 已存在的词（幂等）

    Returns:
        ImportStats: imported / skipped / failed / tag_distribution
    """
    start_time = time.perf_counter()
    target_csv_path = Path(csv_path)
    target_db_path = Path(db_path)
    target_tags = _target_tags(tag_filter)
    tag_counter: Counter[str] = Counter()
    imported = 0
    skipped = 0
    failed = 0
    pending_writes = 0

    _configure_csv_field_size_limit()

    with (
        target_csv_path.open("r", encoding="utf-8", newline="") as csv_file,
        sqlite3.connect(target_db_path) as conn,
    ):
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        _ensure_vocab_tables(conn)

        reader = csv.reader(csv_file)
        progress = _build_progress()
        with progress:
            task_id = progress.add_task("Importing ECDICT", total=None)
            for row_number, row in enumerate(reader, start=1):
                progress.advance(task_id)
                if row_number == 1 and _is_header(row):
                    continue

                try:
                    entry = _parse_row(row, target_tags)
                    if entry is None:
                        skipped += 1
                        continue

                    if _insert_entry(conn, entry, skip_existing=skip_existing):
                        imported += 1
                        pending_writes += 1
                        tag_counter.update(entry.tags)
                    else:
                        skipped += 1

                    if pending_writes >= BATCH_SIZE:
                        conn.commit()
                        pending_writes = 0
                except Exception:
                    failed += 1
                    logger.exception("Failed to import ECDICT row {}", row_number)

        if pending_writes:
            conn.commit()

    return ImportStats(
        imported=imported,
        skipped=skipped,
        failed=failed,
        tag_distribution=dict(tag_counter),
        duration_seconds=time.perf_counter() - start_time,
    )


def _format_tag_distribution(tag_distribution: dict[str, int]) -> str:
    """Format tag counts for CLI output."""
    if not tag_distribution:
        return "none"

    sorted_tags = sorted(tag_distribution.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{tag}: {count}" for tag, count in sorted_tags)


@app.command()
def main(
    csv_path: Annotated[
        Path,
        typer.Option(
            "--csv-path",
            help="ECDICT CSV path.",
        ),
    ] = DEFAULT_CSV_PATH,
    db_path: Annotated[
        Path,
        typer.Option(
            "--db-path",
            help="SQLite database path.",
        ),
    ] = DEFAULT_DB_PATH,
    tags: Annotated[
        str | None,
        typer.Option(
            "--tags",
            help="Comma-separated tag filter, for example ielts,gre,toefl.",
        ),
    ] = None,
    reset: Annotated[
        bool,
        typer.Option(
            "--reset",
            help="Delete all rows from words and mastery before importing.",
        ),
    ] = False,
) -> None:
    """Import exam-relevant ECDICT words into SQLite."""
    console = Console()
    if reset:
        confirmed = typer.confirm(
            f"Delete all rows from words/mastery in {db_path} before import?",
            default=False,
        )
        if not confirmed:
            raise typer.Abort()
        reset_vocab_tables(db_path)

    stats = import_ecdict(
        csv_path=csv_path,
        db_path=db_path,
        tag_filter=_parse_cli_tags(tags),
    )
    console.print(f"Imported {stats.imported} words from ECDICT")
    console.print(f"Skipped: {stats.skipped}")
    console.print(f"Failed: {stats.failed}")
    console.print(f"Tag distribution: {_format_tag_distribution(stats.tag_distribution)}")
    console.print(f"Duration: {stats.duration_seconds:.2f}s")


if __name__ == "__main__":
    app()
