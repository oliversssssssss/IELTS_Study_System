"""Unit tests for src/ielts/vocab/mastery.py.

每个测试用 tmp_path + init_db 建一个全新数据库，直接插入少量 words/mastery 行做夹具，
不触碰真实 db/ielts.sqlite，也不调用任何外部服务。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.ielts import vocab
from src.ielts.schema.init_db import init_db
from src.ielts.vocab import mastery as mastery_module
from src.ielts.vocab.mastery import MasteryEngine


def _seed_word(db_path: Path, lemma: str, status: str = "unknown") -> None:
    """插入一个 word + 对应 mastery 行（状态可指定）。"""
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO words (lemma) VALUES (?)", (lemma,))
        word_id = conn.execute("SELECT id FROM words WHERE lemma = ?", (lemma,)).fetchone()[0]
        conn.execute(
            "INSERT INTO mastery (word_id, status) VALUES (?, ?)",
            (word_id, status),
        )


def _counters(db_path: Path, lemma: str) -> sqlite3.Row:
    """读回某 lemma 的整行 mastery（含计数器）。"""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            """
            SELECT mastery.* FROM mastery
            JOIN words ON words.id = mastery.word_id
            WHERE words.lemma = ?
            """,
            (lemma,),
        ).fetchone()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """初始化一个空的雅思库并预置几个词。"""
    path = tmp_path / "ielts.sqlite"
    init_db(path)
    _seed_word(path, "ambitious")
    _seed_word(path, "study")
    _seed_word(path, "environment")
    return path


@pytest.fixture
def engine(db_path: Path) -> MasteryEngine:
    return MasteryEngine(db_path)


# --------------------------------------------------------------------------- #
# 构造 / 路径
# --------------------------------------------------------------------------- #
def test_init_stores_path_without_connecting(tmp_path: Path) -> None:
    # 指向一个不存在的文件也不应在构造时报错（不做重操作）。
    eng = MasteryEngine(tmp_path / "nope.sqlite")
    assert eng.db_path == tmp_path / "nope.sqlite"


def test_default_db_path_is_package_constant() -> None:
    eng = MasteryEngine()
    assert eng.db_path == mastery_module.DEFAULT_DB_PATH


# --------------------------------------------------------------------------- #
# get_status
# --------------------------------------------------------------------------- #
def test_get_status_hit(engine: MasteryEngine) -> None:
    assert engine.get_status("ambitious") == "unknown"


def test_get_status_case_insensitive(engine: MasteryEngine) -> None:
    assert engine.get_status("Ambitious") == "unknown"


def test_get_status_miss_returns_none(engine: MasteryEngine) -> None:
    assert engine.get_status("serendipity") is None


# --------------------------------------------------------------------------- #
# lookup_batch
# --------------------------------------------------------------------------- #
def test_lookup_batch_dedups_and_skips_misses(engine: MasteryEngine) -> None:
    result = engine.lookup_batch(["ambitious", "ambitious", "study", "ghostword"])
    assert set(result) == {"ambitious", "study"}
    assert result["ambitious"]["status"] == "unknown"


def test_lookup_batch_empty_input(engine: MasteryEngine) -> None:
    assert engine.lookup_batch([]) == {}


def test_lookup_batch_single_query(engine: MasteryEngine, mocker) -> None:
    # 验证只发了一条 SELECT（性能保证：批查不是 N 次查询）。
    # sqlite3.Connection.execute 是只读 C 属性，无法 patch，用 set_trace_callback 计数。
    real_connect = engine._connect
    selects: list[str] = []

    def _wrap() -> sqlite3.Connection:
        conn = real_connect()
        conn.set_trace_callback(
            lambda sql: selects.append(sql) if sql.strip().upper().startswith("SELECT") else None
        )
        return conn

    mocker.patch.object(engine, "_connect", side_effect=_wrap)
    engine.lookup_batch(["ambitious", "study", "environment"])
    assert len(selects) == 1


# --------------------------------------------------------------------------- #
# 计数器
# --------------------------------------------------------------------------- #
def test_record_encounter_reading_increments_both(engine: MasteryEngine, db_path: Path) -> None:
    engine.record_encounter("study", source_type="reading")
    row = _counters(db_path, "study")
    assert row["encounter_count"] == 1
    assert row["reading_encounter"] == 1
    assert row["first_seen_at"] is not None
    assert row["last_seen_at"] is not None


def test_record_encounter_non_reading_only_total(engine: MasteryEngine, db_path: Path) -> None:
    engine.record_encounter("study", source_type="speaking")
    row = _counters(db_path, "study")
    assert row["encounter_count"] == 1
    assert row["reading_encounter"] == 0


def test_record_writing_use_correct_and_wrong(engine: MasteryEngine, db_path: Path) -> None:
    engine.record_writing_use("study", correct=True)
    engine.record_writing_use("study", correct=False)
    row = _counters(db_path, "study")
    assert row["writing_attempt"] == 2
    assert row["writing_correct"] == 1


def test_record_anki_result_success_and_lapse(engine: MasteryEngine, db_path: Path) -> None:
    engine.record_anki_result("study", success=True)
    engine.record_anki_result("study", success=False)
    row = _counters(db_path, "study")
    assert row["anki_correct_count"] == 1
    assert row["anki_lapse_count"] == 1


# --------------------------------------------------------------------------- #
# 状态机升级
# --------------------------------------------------------------------------- #
def test_unknown_does_not_auto_upgrade_on_reading(engine: MasteryEngine) -> None:
    # unknown 永不因计数自动升级，即使读够 5 次。
    for _ in range(5):
        engine.record_encounter("study", source_type="reading")
    assert engine.get_status("study") == "unknown"


def test_add_to_anki_queue_unknown_to_learning(engine: MasteryEngine) -> None:
    assert engine.add_to_anki_queue("study") is True
    assert engine.get_status("study") == "learning"


def test_add_to_anki_queue_already_queued_returns_false(engine: MasteryEngine) -> None:
    engine.add_to_anki_queue("study")
    assert engine.add_to_anki_queue("study") is False


def test_add_to_anki_queue_missing_word_returns_false(engine: MasteryEngine) -> None:
    assert engine.add_to_anki_queue("ghostword") is False


def test_learning_to_familiar_via_reading(engine: MasteryEngine) -> None:
    engine.add_to_anki_queue("study")  # → learning
    for _ in range(5):
        engine.record_encounter("study", source_type="reading")
    assert engine.get_status("study") == "familiar"


def test_learning_to_familiar_via_anki(engine: MasteryEngine) -> None:
    engine.add_to_anki_queue("study")  # → learning
    for _ in range(3):
        engine.record_anki_result("study", success=True)
    assert engine.get_status("study") == "familiar"


def test_familiar_to_mastered_via_writing(engine: MasteryEngine, db_path: Path) -> None:
    _seed_word(db_path, "lucid", status="familiar")
    engine.record_writing_use("lucid", correct=True)
    assert engine.get_status("lucid") == "mastered"


def test_familiar_to_mastered_via_anki(engine: MasteryEngine, db_path: Path) -> None:
    _seed_word(db_path, "lucid", status="familiar")
    for _ in range(10):
        engine.record_anki_result("lucid", success=True)
    assert engine.get_status("lucid") == "mastered"


def test_chained_upgrade_in_single_call(engine: MasteryEngine, db_path: Path) -> None:
    # 词在加入队列前已攒够 reading(≥5) 和 writing(≥1)；一次 add_to_anki_queue 应链式升到 mastered。
    _seed_word(db_path, "eloquent", status="unknown")
    for _ in range(5):
        engine.record_encounter("eloquent", source_type="reading")
    # writing 在 unknown 下不升级，但计数器累积。
    engine.record_writing_use("eloquent", correct=True)
    assert engine.get_status("eloquent") == "unknown"
    assert engine.add_to_anki_queue("eloquent") is True
    assert engine.get_status("eloquent") == "mastered"


def test_mastered_does_not_downgrade(engine: MasteryEngine, db_path: Path) -> None:
    _seed_word(db_path, "lucid", status="mastered")
    for _ in range(10):
        engine.record_anki_result("lucid", success=False)
        engine.record_encounter("lucid", source_type="reading")
    assert engine.get_status("lucid") == "mastered"


# --------------------------------------------------------------------------- #
# marked 保护
# --------------------------------------------------------------------------- #
def test_marked_is_protected_from_auto_upgrade(engine: MasteryEngine, db_path: Path) -> None:
    _seed_word(db_path, "ineffable", status="marked")
    for _ in range(10):
        engine.record_encounter("ineffable", source_type="reading")
        engine.record_anki_result("ineffable", success=True)
        engine.record_writing_use("ineffable", correct=True)
    assert engine.get_status("ineffable") == "marked"


# --------------------------------------------------------------------------- #
# auto_add_unknown
# --------------------------------------------------------------------------- #
def test_auto_add_unknown_false_ignores(engine: MasteryEngine) -> None:
    engine.record_encounter("neologism", source_type="reading", auto_add_unknown=False)
    assert engine.get_status("neologism") is None


def test_auto_add_unknown_true_creates_word(engine: MasteryEngine) -> None:
    engine.record_encounter("neologism", source_type="reading", auto_add_unknown=True)
    assert engine.get_status("neologism") == "unknown"


# --------------------------------------------------------------------------- #
# 标记 / 手动 / note id / 幂等
# --------------------------------------------------------------------------- #
def test_mark_priority_sets_flag_without_status_change(
    engine: MasteryEngine, db_path: Path
) -> None:
    engine.mark_priority("study", priority=True)
    row = _counters(db_path, "study")
    assert row["is_priority"] == 1
    assert row["status"] == "unknown"
    engine.mark_priority("study", priority=False)
    assert _counters(db_path, "study")["is_priority"] == 0


def test_add_to_anki_queue_with_priority_sets_flag(engine: MasteryEngine, db_path: Path) -> None:
    assert engine.add_to_anki_queue("study", priority=True) is True
    assert _counters(db_path, "study")["is_priority"] == 1


def test_manual_mark_forces_status(engine: MasteryEngine) -> None:
    engine.manual_mark("study", "marked")
    assert engine.get_status("study") == "marked"


def test_manual_mark_is_idempotent(engine: MasteryEngine, db_path: Path) -> None:
    engine.manual_mark("study", "mastered")
    engine.manual_mark("study", "mastered")
    assert engine.get_status("study") == "mastered"


def test_manual_mark_rejects_invalid_status(engine: MasteryEngine) -> None:
    with pytest.raises(ValueError, match="非法状态"):
        engine.manual_mark("study", "wizard")  # type: ignore[arg-type]


def test_set_anki_note_id(engine: MasteryEngine, db_path: Path) -> None:
    engine.set_anki_note_id("study", 123456)
    assert _counters(db_path, "study")["anki_note_id"] == 123456


# --------------------------------------------------------------------------- #
# 阈值可改 + 未知词鲁棒性
# --------------------------------------------------------------------------- #
def test_threshold_is_configurable(engine: MasteryEngine, monkeypatch) -> None:
    monkeypatch.setattr(mastery_module, "LEARNING_TO_FAMILIAR_READING_THRESHOLD", 2)
    engine.add_to_anki_queue("study")  # → learning
    engine.record_encounter("study", source_type="reading")
    engine.record_encounter("study", source_type="reading")
    assert engine.get_status("study") == "familiar"


def test_mutations_on_missing_word_do_not_raise(engine: MasteryEngine) -> None:
    # 所有按 lemma 的写方法对不存在的词都应安全 no-op。
    engine.record_encounter("ghostword")
    engine.record_writing_use("ghostword", correct=True)
    engine.record_anki_result("ghostword", success=True)
    engine.mark_priority("ghostword")
    engine.manual_mark("ghostword", "marked")
    engine.set_anki_note_id("ghostword", 1)
    assert engine.get_status("ghostword") is None


def test_vocab_package_importable() -> None:
    # mastery 应可从 vocab 包路径正常导入（无循环依赖）。
    assert hasattr(vocab, "__name__")
