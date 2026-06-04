"""P0 end-to-end smoke test for NLP, LLM, and Bark push."""

from loguru import logger

from src.ielts.llm import chat_structured
from src.ielts.nlp.processor import extract_unique_lemmas
from src.ielts.tools.push import push_bark

SYSTEM_PROMPT = """You write concise English poems and return JSON only.

Return exactly one JSON object with this schema:
{"poem": "four-line English poem"}

Do not include Markdown, commentary, or extra keys.
"""
SMOKE_TEXT = "The students were studying ambitious goals."
LEMMA_LIMIT = 3
POEM_LINES = 4
LLM_MAX_TOKENS = 300
LLM_TEMPERATURE = 0.3
BARK_TITLE = "🎉 P0 完成"
BARK_GROUP = "ielts"


def main() -> None:
    """Run the P0 infrastructure smoke test end to end."""
    logger.info("Starting P0 smoke test.")

    lemma_occurrences = extract_unique_lemmas(SMOKE_TEXT)
    lemmas = list(lemma_occurrences)[:LEMMA_LIMIT]
    if len(lemmas) < LEMMA_LIMIT:
        raise ValueError(f"Expected at least {LEMMA_LIMIT} lemmas, got {len(lemmas)}.")
    logger.info("NLP extracted lemmas: {}", ", ".join(lemmas))

    user_prompt = (
        "Write JSON for a P0 infrastructure smoke test.\n"
        f"Use these lemmas: {', '.join(lemmas)}.\n"
        "Theme: 环境就绪.\n"
        f'Return key "poem" containing exactly {POEM_LINES} English lines.'
    )
    logger.info("Requesting DeepSeek JSON poem.")
    data = chat_structured(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
    )

    poem = data.get("poem")
    if not isinstance(poem, str) or not poem.strip():
        raise ValueError("DeepSeek JSON response must include a non-empty string poem.")
    poem = poem.strip()

    line_count = len([line for line in poem.splitlines() if line.strip()])
    if line_count != POEM_LINES:
        raise ValueError(f"Expected {POEM_LINES} poem lines, got {line_count}.")
    logger.info("DeepSeek wrote a JSON poem.")

    logger.info("Sending Bark push.")
    if not push_bark(BARK_TITLE, poem, group=BARK_GROUP):
        raise RuntimeError("Bark push failed.")
    logger.info("Bark push delivered.")

    logger.info("✅ P0 smoke test passed:\n{}", poem)


if __name__ == "__main__":
    main()
