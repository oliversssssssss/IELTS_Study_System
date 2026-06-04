"""SQLite database initialization for the IELTS study system."""

import re
import sqlite3
from pathlib import Path

from loguru import logger

DEFAULT_DB_PATH = Path("db/ielts.sqlite")
SCHEMA_DIR = Path(__file__).resolve().parent
SCHEMA_FILE_PATTERN = re.compile(r"^\d{2}.*\.sql$")


def _schema_files(schema_dir: Path | None = None) -> list[Path]:
    """Return ordered pre-v1 schema files from the schema directory."""
    target_dir = SCHEMA_DIR if schema_dir is None else schema_dir
    if not target_dir.exists():
        raise FileNotFoundError(f"Schema directory does not exist: {target_dir}")
    if not target_dir.is_dir():
        raise FileNotFoundError(f"Schema path is not a directory: {target_dir}")

    return sorted(
        path
        for path in target_dir.iterdir()
        if path.is_file() and SCHEMA_FILE_PATTERN.match(path.name)
    )


def _table_count(conn: sqlite3.Connection) -> int:
    """Count user tables in the active SQLite connection."""
    row = conn.execute(
        """
        SELECT COUNT(*) AS table_count
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        """
    ).fetchone()
    return int(row["table_count"])


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """初始化雅思系统数据库。

    执行 src/ielts/schema/ 下所有 *.sql 文件（按字母顺序），
    启用 WAL 模式，幂等可重复执行。

    Args:
        db_path: SQLite 数据库文件路径，默认 db/ielts.sqlite

    Raises:
        sqlite3.Error: 如果 schema 执行失败
        FileNotFoundError: 如果 schema 目录不存在
    """
    target_path = Path(db_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    schema_files = _schema_files()
    with sqlite3.connect(target_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        journal_mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]

        for schema_file in schema_files:
            logger.info("Executing schema file: {}", schema_file)
            conn.executescript(schema_file.read_text(encoding="utf-8"))

        logger.info(
            "✅ Database initialized at {} ({} tables, {} enabled)",
            target_path,
            _table_count(conn),
            str(journal_mode).upper(),
        )


if __name__ == "__main__":
    init_db()
