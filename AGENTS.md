# IELTS Prep System

Personal IELTS preparation tool. Single user. Not a product. Open-source for reference only.

## Stack

- Python 3.11, managed by `uv` (NEVER use pip)
- SQLite via `sqlite3` stdlib (no ORM, hand-written SQL)
- DeepSeek API (OpenAI-compatible) for LLM
- spaCy `en_core_web_lg` for NLP
- Tesseract (English) for OCR
- APScheduler for scheduling
- FastAPI when an HTTP endpoint is needed
- pytest for tests, ruff for lint+format
- Loguru for logging
- AnkiConnect HTTP API (port 8765) for Anki integration
- Server酱 (WeChat push), Feishu webhook for notifications

## Layout
src/ielts/ # all code lives here
llm.py # DeepSeek client (shared by all modules)
schema/*.sql # database schema (numbered files)
schema/migrations/ # post-v1 schema changes
nlp/ # spaCy + Tesseract wrappers
vocab/ # vocabulary module
reading/ # reading module
writing/ # writing module
speaking/ # speaking module
tools/ # push, anki, common helpers
api/ # FastAPI endpoints
cli.py # typer CLI entrypoint
scheduler.py # APScheduler setup
tests/ # mirrors src/ielts/ structure
db/ # SQLite files (NEVER manually edit)
data/ # datasets (gitignored)
vault/雅思/ # Obsidian output (markdown)
config/ # *.yaml configs; secrets.env is read-only
.tasks/ # task briefs from project manager
docs/CHANGELOG.md # append one line per completed task

## Commands

| Action | Command |
|---|---|
| Run module | `uv run python -m src.ielts.<module>` |
| Add package | `uv add <pkg>` |
| Add dev pkg | `uv add --dev <pkg>` |
| Run tests | `uv run pytest tests/ -v` |
| Lint check | `uv run ruff check src/` |
| Lint fix | `uv run ruff check --fix src/` |
| Format | `uv run ruff format src/` |
| DB inspect | `sqlite3 db/ielts.sqlite "SELECT ..."` |

## Hard Rules

1. NEVER use `print()`. Use `from loguru import logger`.
2. NEVER use bare `except:` or `except Exception: pass`. Log with `logger.exception`.
3. NEVER read or modify `config/secrets.env` for output/logs.
4. NEVER hardcode API keys, paths, or magic numbers.
5. NEVER edit files under `vault/70_我的笔记/`.
6. NEVER add a dependency >50MB without approval in the task brief.
7. NEVER refactor code outside the task brief's scope.
8. NEVER claim "DONE" without running the validation in the task brief.
9. NEVER guess file paths or API signatures. If unsure, output `🛑 BLOCKED:`.
10. NEVER use global mutable state; use `functools.lru_cache` if caching is needed.

## Code Style

- Type hints required on all function signatures
- Google-style docstrings on public functions
- Chinese OK in comments and string literals; English only for identifiers
- f-strings only (never `%` or `.format()`)
- `pathlib.Path` for all path handling
- Timezone-aware datetimes: `datetime.now(ZoneInfo("Asia/Shanghai"))`

## LLM Call Pattern

All LLM calls go through `src/ielts/llm.py`. The system prompt MUST be a module-level constant (no f-string variables) so DeepSeek prompt caching works.

```python
SYSTEM_PROMPT = """<immutable text>""" # module-level, no variables

def my_task(user_input: str) -> dict:
 return chat_structured(
 system=SYSTEM_PROMPT,
 user=f"<dynamic content with {user_input}>",
 max_tokens=2000,
 temperature=0.3,
 )
```

## Database Rules

- Schema files in `src/ielts/schema/*.sql` are source of truth pre-v1
- After v1, all changes go to `src/ielts/schema/migrations/YYYYMMDD_NN_desc.sql`
- Never edit a committed migration
- Code touching `mastery` table MUST use `MasteryEngine` class
- Use parameterized queries always
- Use `with sqlite3.connect(...) as conn:` context manager
- Set `conn.row_factory = sqlite3.Row` for dict-like access

## Testing Rules

- New module → `tests/test_<module>.py` MUST exist
- Mock all external services (LLM, AnkiConnect, web fetches)
- Coverage target: 70%+ for new code
- Tests must complete in <30 seconds total

## Definition of Done

A task is complete only when ALL are true:

1. `uv run ruff check src/` exits 0
2. `uv run ruff format src/` has been run
3. `uv run pytest tests/ -v` exits 0
4. The manual validation in the task brief produces expected output
5. `docs/CHANGELOG.md` has a new one-line entry
6. Final output is exactly: `✅ DONE: <one-sentence summary>`
7. Anything intentionally deferred is listed under `🚧 SKIPPED:`

## When Stuck

Do NOT guess. Output this block and STOP:
🛑 BLOCKED
PROBLEM: <specific issue in 1-2 sentences>
TRIED: <what you attempted, 1-3 bullets>
NEED: <specific information or decision required>

## Known Pitfalls

- spaCy load is slow — cache as module-level singleton in `nlp/processor.py`
- AnkiConnect from WSL2: use IP from `cat /etc/resolv.conf`, not `localhost`
- DeepSeek base URL is `https://api.deepseek.com/v1` (NOT `.ai`)
- DeepSeek JSON mode: set `response_format={"type": "json_object"}` AND mention JSON in prompt
- Tesseract for IELTS reading: use `--psm 6 --oem 3` for single-column text
- SQLite concurrency: enable WAL at DB creation: `conn.execute("PRAGMA journal_mode=WAL")`
- Configure Loguru once at app startup, not in each module
- Working directory is on Windows mount (/mnt/f/...), file I/O is 5-10x slower than native ext4

## Long Tasks

For tasks expected to take >30 minutes:

```bash
codex cloud submit --task "<full brief>" --branch <branch-name>
```

## Cheap Mode

For obvious formatting/renaming/docstring tasks:

```bash
codex --model gpt-5-codex-mini "<task>"
```

## Commit Format

Prepare commit message after each task (user runs `git commit`):
<type>(<scope>): <description>
detail 1
detail 2
Task: <task-id from brief>
Types: `feat` `fix` `refactor` `test` `docs` `chore`
Scopes: `vocab` `reading` `writing` `speaking` `nlp` `db` `infra` `tools`
