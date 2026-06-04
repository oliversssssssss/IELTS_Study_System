"""词汇熟练度引擎（业务层中枢）。

`MasteryEngine` 是所有"熟练度"读写的统一入口：阅读遇词、Anki 对错、写作用词都经过它，
累积到阈值后状态机自动升级。后续 reading / writing / speaking / anki 模块都 import 本类。

设计要点（详见 docs/CHANGELOG 与任务 P1-05 plan）：
- 每个方法新建连接（线程安全；WAL 建库时已开，并发读无锁竞争），__init__ 不做重操作。
- lemma 存在 `words` 表（COLLATE NOCASE），`mastery` 主键是 word_id，按 lemma 读写需先解析。
- 状态机阈值是模块级常量（不硬编码、可被测试覆写）；marked 永不自动变，unknown 不自动升级。
- 本模块只依赖 stdlib + loguru，绝不 import 兄弟业务模块（避免循环依赖）。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal

from loguru import logger

MasteryStatus = Literal["unknown", "learning", "familiar", "mastered", "marked"]
SourceType = Literal["reading", "writing_mine", "writing_sample", "speaking", "listening"]

DEFAULT_DB_PATH = Path("db/ielts.sqlite")

# 状态机阈值（不硬编码；测试可 monkeypatch 本模块常量即时生效，因函数内按全局名查找）。
LEARNING_TO_FAMILIAR_READING_THRESHOLD = 5
LEARNING_TO_FAMILIAR_ANKI_THRESHOLD = 3
FAMILIAR_TO_MASTERED_WRITING_THRESHOLD = 1
FAMILIAR_TO_MASTERED_ANKI_THRESHOLD = 10

# 合法状态集合，供 manual_mark 校验（与 schema CHECK 约束一致）。
_VALID_STATUSES: frozenset[str] = frozenset(
    ("unknown", "learning", "familiar", "mastered", "marked")
)


def _next_status(row: sqlite3.Row) -> MasteryStatus:
    """根据计数器计算当前行的下一目标状态（单步转移）。

    marked / unknown / mastered 不自动改变；learning / familiar 按阈值前进。
    阈值通过模块全局名查找读取，便于测试覆写。

    Args:
        row: 一行 mastery 记录（需含 status 及各计数器）。

    Returns:
        转移后的目标状态；若不该自动变则原样返回。
    """
    status: str = row["status"]

    if status == "learning":
        if (
            row["reading_encounter"] >= LEARNING_TO_FAMILIAR_READING_THRESHOLD
            or row["anki_correct_count"] >= LEARNING_TO_FAMILIAR_ANKI_THRESHOLD
        ):
            return "familiar"
        return "learning"

    if status == "familiar":
        if (
            row["writing_correct"] >= FAMILIAR_TO_MASTERED_WRITING_THRESHOLD
            or row["anki_correct_count"] >= FAMILIAR_TO_MASTERED_ANKI_THRESHOLD
        ):
            return "mastered"
        return "familiar"

    # marked（最高优先级，自动升级忽略）、unknown（仅 add_to_anki_queue 显式驱动）、
    # mastered（终态，不自动降级）均原样返回。
    return status  # type: ignore[return-value]


class MasteryEngine:
    """词汇熟练度的统一读写入口。

    线程安全：每个方法独立开关连接，不持有跨调用的连接对象。
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        """记录数据库路径（不在构造时连接或做重操作）。

        Args:
            db_path: SQLite 数据库路径，默认 db/ielts.sqlite。
        """
        self.db_path = Path(db_path)

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _connect(self) -> sqlite3.Connection:
        """打开一个配置好的连接（Row 工厂 + 外键约束）。

        WAL 为库级持久属性，建库时已开启，无需每连接重设；外键是连接级，每次都设。
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @staticmethod
    def _resolve_word_id(conn: sqlite3.Connection, lemma: str) -> int | None:
        """把 lemma 解析为 word_id（COLLATE NOCASE 大小写不敏感）。"""
        row = conn.execute("SELECT id FROM words WHERE lemma = ?", (lemma,)).fetchone()
        return int(row["id"]) if row is not None else None

    @staticmethod
    def _ensure_word(conn: sqlite3.Connection, lemma: str) -> int:
        """确保 lemma 在 words/mastery 中存在（用于 auto_add_unknown），返回 word_id。"""
        conn.execute("INSERT OR IGNORE INTO words (lemma) VALUES (?)", (lemma,))
        row = conn.execute("SELECT id FROM words WHERE lemma = ?", (lemma,)).fetchone()
        word_id = int(row["id"])
        conn.execute(
            "INSERT OR IGNORE INTO mastery (word_id, status) VALUES (?, 'unknown')",
            (word_id,),
        )
        return word_id

    @staticmethod
    def _recompute(conn: sqlite3.Connection, word_id: int) -> None:
        """在同一事务内回读并应用状态机，直至状态稳定（支持一次链式升级）。"""
        while True:
            row = conn.execute("SELECT * FROM mastery WHERE word_id = ?", (word_id,)).fetchone()
            if row is None:
                return
            target = _next_status(row)
            if target == row["status"]:
                return
            conn.execute(
                """
                UPDATE mastery
                SET status = ?, last_status_change = CURRENT_TIMESTAMP
                WHERE word_id = ?
                """,
                (target, word_id),
            )

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    def get_status(self, lemma: str) -> MasteryStatus | None:
        """返回某 lemma 的熟练度状态；词不存在时返回 None。

        Args:
            lemma: 词元（大小写不敏感）。

        Returns:
            状态字符串，或 None（词典中无此词）。
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT mastery.status AS status
                FROM mastery
                JOIN words ON words.id = mastery.word_id
                WHERE words.lemma = ?
                """,
                (lemma,),
            ).fetchone()
        return row["status"] if row is not None else None

    def lookup_batch(self, lemmas: list[str]) -> dict[str, dict]:
        """一次性批量查询多个 lemma 的完整 mastery 行。

        用单条 `WHERE lemma IN (...)` 查询。输入会去重；未命中的 lemma 不出现在结果中。
        键为数据库中的规范 lemma（COLLATE NOCASE 命中）。

        Args:
            lemmas: 词元列表（可含重复）。

        Returns:
            映射 {lemma: {该词的全部 mastery 字段 + lemma}}。
        """
        unique = list(dict.fromkeys(lemmas))
        if not unique:
            return {}

        placeholders = ",".join("?" * len(unique))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT words.lemma AS lemma, mastery.*
                FROM mastery
                JOIN words ON words.id = mastery.word_id
                WHERE words.lemma IN ({placeholders})
                """,
                unique,
            ).fetchall()
        return {row["lemma"]: dict(row) for row in rows}

    # ------------------------------------------------------------------ #
    # 计数器更新（均在单事务内：更新 → recompute）
    # ------------------------------------------------------------------ #
    def record_encounter(
        self,
        lemma: str,
        source_type: SourceType = "reading",
        *,
        auto_add_unknown: bool = False,
    ) -> None:
        """记录一次"遇到该词"，递增计数并尝试自动升级。

        `encounter_count` 总是自增；`source_type == 'reading'` 时额外自增
        `reading_encounter`。更新首/末次见到时间。

        Args:
            lemma: 词元。
            source_type: 来源类型，默认 reading。
            auto_add_unknown: 词不在词典时是否自动建最小词条；False（默认）则静默忽略。
        """
        with self._connect() as conn:
            word_id = self._resolve_word_id(conn, lemma)
            if word_id is None:
                if not auto_add_unknown:
                    logger.debug("record_encounter 跳过未知词（不在词典）。")
                    return
                word_id = self._ensure_word(conn, lemma)

            reading_delta = 1 if source_type == "reading" else 0
            conn.execute(
                """
                UPDATE mastery
                SET encounter_count = encounter_count + 1,
                    reading_encounter = reading_encounter + ?,
                    first_seen_at = COALESCE(first_seen_at, CURRENT_TIMESTAMP),
                    last_seen_at = CURRENT_TIMESTAMP
                WHERE word_id = ?
                """,
                (reading_delta, word_id),
            )
            self._recompute(conn, word_id)

    def record_writing_use(self, lemma: str, *, correct: bool) -> None:
        """记录一次写作中使用该词。

        `writing_attempt` 总是自增；`correct=True` 时额外自增 `writing_correct`。

        Args:
            lemma: 词元。
            correct: 是否用对。
        """
        with self._connect() as conn:
            word_id = self._resolve_word_id(conn, lemma)
            if word_id is None:
                logger.debug("record_writing_use 跳过未知词（不在词典）。")
                return
            conn.execute(
                """
                UPDATE mastery
                SET writing_attempt = writing_attempt + 1,
                    writing_correct = writing_correct + ?
                WHERE word_id = ?
                """,
                (1 if correct else 0, word_id),
            )
            self._recompute(conn, word_id)

    def record_anki_result(self, lemma: str, *, success: bool) -> None:
        """记录一次 Anki 复习结果。

        `success=True` 自增 `anki_correct_count`，否则自增 `anki_lapse_count`。

        Args:
            lemma: 词元。
            success: 本次复习是否答对。
        """
        with self._connect() as conn:
            word_id = self._resolve_word_id(conn, lemma)
            if word_id is None:
                logger.debug("record_anki_result 跳过未知词（不在词典）。")
                return
            if success:
                conn.execute(
                    "UPDATE mastery SET anki_correct_count = anki_correct_count + 1 "
                    "WHERE word_id = ?",
                    (word_id,),
                )
            else:
                conn.execute(
                    "UPDATE mastery SET anki_lapse_count = anki_lapse_count + 1 WHERE word_id = ?",
                    (word_id,),
                )
            self._recompute(conn, word_id)

    # ------------------------------------------------------------------ #
    # 状态/标记操作
    # ------------------------------------------------------------------ #
    def add_to_anki_queue(self, lemma: str, *, priority: bool = False) -> bool:
        """把词加入 Anki 学习队列：唯一能把 unknown 升到 learning 的入口。

        Args:
            lemma: 词元。
            priority: 是否同时置高优先级标记。

        Returns:
            True 表示本次新加入（unknown→learning）；False 表示词不存在或已在队列/更高状态。
        """
        with self._connect() as conn:
            word_id = self._resolve_word_id(conn, lemma)
            if word_id is None:
                logger.debug("add_to_anki_queue 跳过未知词（不在词典）。")
                return False

            row = conn.execute(
                "SELECT status FROM mastery WHERE word_id = ?", (word_id,)
            ).fetchone()
            if priority:
                conn.execute("UPDATE mastery SET is_priority = 1 WHERE word_id = ?", (word_id,))

            if row["status"] != "unknown":
                return False

            conn.execute(
                """
                UPDATE mastery
                SET status = 'learning', last_status_change = CURRENT_TIMESTAMP
                WHERE word_id = ?
                """,
                (word_id,),
            )
            # 计数器若已达阈值，可链式继续升级。
            self._recompute(conn, word_id)
            return True

    def mark_priority(self, lemma: str, *, priority: bool = True) -> None:
        """设置/清除优先级标记，不改变熟练度状态。

        Args:
            lemma: 词元。
            priority: True 置位，False 清除。
        """
        with self._connect() as conn:
            word_id = self._resolve_word_id(conn, lemma)
            if word_id is None:
                logger.debug("mark_priority 跳过未知词（不在词典）。")
                return
            conn.execute(
                "UPDATE mastery SET is_priority = ? WHERE word_id = ?",
                (1 if priority else 0, word_id),
            )

    def manual_mark(self, lemma: str, status: MasteryStatus) -> None:
        """手动强制设置状态（绕过状态机），用于 marked 或人工纠正。

        Args:
            lemma: 词元。
            status: 目标状态，必须是合法值之一。

        Raises:
            ValueError: status 非法。
        """
        if status not in _VALID_STATUSES:
            raise ValueError(f"非法状态：{status!r}")
        with self._connect() as conn:
            word_id = self._resolve_word_id(conn, lemma)
            if word_id is None:
                logger.debug("manual_mark 跳过未知词（不在词典）。")
                return
            conn.execute(
                """
                UPDATE mastery
                SET status = ?, last_status_change = CURRENT_TIMESTAMP
                WHERE word_id = ?
                """,
                (status, word_id),
            )

    def set_anki_note_id(self, lemma: str, note_id: int) -> None:
        """回填 Anki 卡片 note id（供 AnkiConnect 建卡后调用）。

        Args:
            lemma: 词元。
            note_id: AnkiConnect 返回的 note id。
        """
        with self._connect() as conn:
            word_id = self._resolve_word_id(conn, lemma)
            if word_id is None:
                logger.debug("set_anki_note_id 跳过未知词（不在词典）。")
                return
            conn.execute(
                "UPDATE mastery SET anki_note_id = ? WHERE word_id = ?",
                (note_id, word_id),
            )
