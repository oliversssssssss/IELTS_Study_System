"""Unit tests for src/ielts/llm.py — 全部 mock，绝不调用真实 API。

mock 策略：patch `src.ielts.llm._get_client` 返回一个假客户端，其
`.chat.completions.create` 返回构造好的假响应。由于 `_get_client` 被整体替换，
`lru_cache` 自然失效，测试间无缓存污染。
"""

import json
from types import SimpleNamespace

import pytest

from src.ielts import llm
from src.ielts.llm import (
    LLMConfigError,
    LLMError,
    LLMJSONError,
    LLMResponseError,
    chat,
    chat_structured,
    embed,
)


def _fake_response(content: str | None) -> SimpleNamespace:
    """构造形如 openai 响应的最小对象：response.choices[0].message.content。"""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


@pytest.fixture
def fake_client(mocker):
    """提供假客户端并 patch `_get_client` 返回它。"""
    client = mocker.MagicMock()
    mocker.patch("src.ielts.llm._get_client", return_value=client)
    return client


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #
def test_chat_returns_content(fake_client) -> None:
    fake_client.chat.completions.create.return_value = _fake_response("OK")
    result = chat([{"role": "user", "content": "hi"}], max_tokens=10)
    assert result == "OK"


def test_chat_passes_expected_kwargs(fake_client) -> None:
    fake_client.chat.completions.create.return_value = _fake_response("OK")
    messages = [{"role": "user", "content": "hi"}]
    chat(messages, model="deepseek-chat", temperature=0.5, max_tokens=42, timeout=12.0)

    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "deepseek-chat"
    assert kwargs["messages"] == messages
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_tokens"] == 42
    assert kwargs["timeout"] == 12.0
    # chat 不应设置 response_format。
    assert "response_format" not in kwargs


def test_chat_raises_on_empty_content(fake_client) -> None:
    fake_client.chat.completions.create.return_value = _fake_response("")
    with pytest.raises(LLMResponseError):
        chat([{"role": "user", "content": "hi"}])


def test_chat_raises_on_none_content(fake_client) -> None:
    fake_client.chat.completions.create.return_value = _fake_response(None)
    with pytest.raises(LLMResponseError):
        chat([{"role": "user", "content": "hi"}])


def test_chat_raises_on_empty_messages(fake_client) -> None:
    with pytest.raises(LLMError):
        chat([])
    fake_client.chat.completions.create.assert_not_called()


# --------------------------------------------------------------------------- #
# chat_structured
# --------------------------------------------------------------------------- #
def test_chat_structured_returns_dict(fake_client) -> None:
    fake_client.chat.completions.create.return_value = _fake_response('{"ok": true}')
    result = chat_structured("You output JSON only.", "Output JSON.")
    assert result == {"ok": True}


def test_chat_structured_sets_json_response_format(fake_client) -> None:
    fake_client.chat.completions.create.return_value = _fake_response('{"ok": true}')
    chat_structured("sys JSON", "user JSON")

    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    # system / user 应被组装为两条消息。
    roles = [m["role"] for m in kwargs["messages"]]
    assert roles == ["system", "user"]


def test_chat_structured_raises_on_invalid_json(fake_client) -> None:
    fake_client.chat.completions.create.return_value = _fake_response("not json{")
    with pytest.raises(LLMJSONError):
        chat_structured("sys JSON", "user JSON")


def test_chat_structured_raises_on_non_object_json(fake_client) -> None:
    # 合法 JSON，但是数组而非对象。
    fake_client.chat.completions.create.return_value = _fake_response("[1, 2, 3]")
    with pytest.raises(LLMJSONError):
        chat_structured("sys JSON", "user JSON")


def test_llmjson_is_subclass_of_response_error() -> None:
    # 保证调用方可用 LLMResponseError / LLMError 统一兜底。
    assert issubclass(LLMJSONError, LLMResponseError)
    assert issubclass(LLMResponseError, LLMError)


# --------------------------------------------------------------------------- #
# embed
# --------------------------------------------------------------------------- #
def test_embed_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        embed(["hello"])


# --------------------------------------------------------------------------- #
# _get_client（配置与缓存）
# --------------------------------------------------------------------------- #
def test_get_client_missing_key_raises(mocker, monkeypatch) -> None:
    # 阻止 load_dotenv 从真实 secrets.env 注入 key。
    mocker.patch("src.ielts.llm.load_dotenv")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    llm._get_client.cache_clear()
    try:
        with pytest.raises(LLMConfigError):
            llm._get_client()
    finally:
        llm._get_client.cache_clear()


def test_get_client_builds_with_env(mocker, monkeypatch) -> None:
    mocker.patch("src.ielts.llm.load_dotenv")
    fake_openai = mocker.patch("src.ielts.llm.OpenAI")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.test/v1")
    llm._get_client.cache_clear()
    try:
        client = llm._get_client()
        fake_openai.assert_called_once_with(
            api_key="sk-test-key", base_url="https://example.test/v1"
        )
        # 第二次调用命中 lru_cache，不再重建。
        assert llm._get_client() is client
        fake_openai.assert_called_once()
    finally:
        llm._get_client.cache_clear()


def test_get_client_defaults_base_url(mocker, monkeypatch) -> None:
    mocker.patch("src.ielts.llm.load_dotenv")
    fake_openai = mocker.patch("src.ielts.llm.OpenAI")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    llm._get_client.cache_clear()
    try:
        llm._get_client()
        kwargs = fake_openai.call_args.kwargs
        assert kwargs["base_url"] == llm.DEFAULT_BASE_URL
    finally:
        llm._get_client.cache_clear()


def test_json_decode_error_does_not_log_response(fake_client, mocker) -> None:
    # 保证 JSON 解析失败时，日志中不出现 response 原文（防泄露）。
    fake_client.chat.completions.create.return_value = _fake_response("SECRET-LEAK{")
    log_spy = mocker.patch("src.ielts.llm.logger.error")
    with pytest.raises(LLMJSONError):
        chat_structured("sys JSON", "user JSON")
    logged = " ".join(str(a) for call in log_spy.call_args_list for a in call.args)
    assert "SECRET-LEAK" not in logged


def test_response_format_omitted_means_plain_text(fake_client) -> None:
    # 回归保护：确保 _complete 在无 response_format 时不会塞入该键。
    fake_client.chat.completions.create.return_value = _fake_response(json.dumps({"ignored": 1}))
    chat([{"role": "user", "content": "hi"}])
    assert "response_format" not in fake_client.chat.completions.create.call_args.kwargs
