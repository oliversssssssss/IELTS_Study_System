-- src/ielts/schema/01_vocab.sql
-- IELTS Vocabulary Schema (P1-01)
--
-- Tables:
-- words - Base dictionary entries (one row per lemma)
-- mastery - Per-user mastery state (1:1 with words)
-- word_occurrences - Trace where each word appears
-- collocations - Word collocations (reserved for v2)
--
-- All tables use IF NOT EXISTS for idempotent execution.

-- ============================================================
-- words
-- ============================================================
CREATE TABLE IF NOT EXISTS words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lemma TEXT UNIQUE NOT NULL COLLATE NOCASE,
    pos TEXT,
    ipa TEXT,
    definitions TEXT,
    cefr_level TEXT,
    tags TEXT,
    bnc_freq INTEGER,
    coca_freq INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_words_lemma ON words(lemma);

-- ============================================================
-- mastery
-- ============================================================
CREATE TABLE IF NOT EXISTS mastery (
    word_id INTEGER PRIMARY KEY REFERENCES words(id),
    status TEXT NOT NULL DEFAULT 'unknown'
        CHECK (status IN ('unknown', 'learning', 'familiar', 'mastered', 'marked')),
    encounter_count INTEGER DEFAULT 0,
    reading_encounter INTEGER DEFAULT 0,
    writing_attempt INTEGER DEFAULT 0,
    writing_correct INTEGER DEFAULT 0,
    anki_note_id INTEGER,
    anki_correct_count INTEGER DEFAULT 0,
    anki_lapse_count INTEGER DEFAULT 0,
    first_seen_at TIMESTAMP,
    last_seen_at TIMESTAMP,
    last_status_change TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_priority BOOLEAN DEFAULT 0,
    user_note TEXT
);

CREATE INDEX IF NOT EXISTS idx_mastery_status ON mastery(status);
CREATE INDEX IF NOT EXISTS idx_mastery_priority ON mastery(is_priority) WHERE is_priority = 1;

-- ============================================================
-- word_occurrences
-- ============================================================
CREATE TABLE IF NOT EXISTS word_occurrences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word_id INTEGER REFERENCES words(id),
    source_type TEXT NOT NULL
        CHECK (source_type IN ('reading', 'writing_mine', 'writing_sample', 'speaking', 'listening')),
    source_id INTEGER,
    sentence TEXT,
    encountered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_occ_word ON word_occurrences(word_id);
CREATE INDEX IF NOT EXISTS idx_occ_source ON word_occurrences(source_type, source_id);

-- ============================================================
-- collocations
-- ============================================================
CREATE TABLE IF NOT EXISTS collocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    head_word_id INTEGER REFERENCES words(id),
    pattern TEXT,
    phrase TEXT NOT NULL,
    example TEXT,
    register TEXT,
    band_level INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_col_head ON collocations(head_word_id);
