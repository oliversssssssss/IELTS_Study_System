"""Smoke tests for src/ielts/llm.py — 调用真实 DeepSeek API。

默认会运行（与 tests/test_p0_05_apis.py 一致），但：
  - 标记为 `@pytest.mark.smoke`，可用 `-m "not smoke"` 排除；
  - 未配置 DEEPSEEK_API_KEY 时自动跳过。

只跑冒烟：  uv run pytest -m smoke
排除冒烟：  uv run pytest -m "not smoke"
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.ielts.llm import chat, chat_structured

# 真实 key 在 secrets.env 里，导入期加载以便 skipif 判断。
load_dotenv(Path(__file__).resolve().parents[1] / "config" / "secrets.env")

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        not os.getenv("DEEPSEEK_API_KEY"),
        reason="DEEPSEEK_API_KEY not set; skipping real-API smoke test",
    ),
]


def test_chat_real_api() -> None:
    """对应 Goal 验证 1：基础调用。"""
    result = chat([{"role": "user", "content": "Reply OK only."}], max_tokens=10)
    assert result


def test_chat_structured_real_api() -> None:
    """对应 Goal 验证 2：JSON 输出。"""
    result = chat_structured("You output JSON only.", 'Output {"ok": true} as JSON.')
    assert result.get("ok") in (True, "true", 1)
