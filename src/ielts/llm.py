"""DeepSeek LLM 客户端封装（基础设施层）。

所有模块（vocab / reading / writing / speaking）的 LLM 调用都必须经过本模块，
不要在别处直接 `from openai import OpenAI`。

公共 API（已固定，请勿改签名）：
    chat(messages, ...) -> str            纯文本对话
    chat_structured(system, user, ...) -> dict   强制 JSON 输出
    embed(texts) -> list[list[float]]     嵌入（P4 才实现，目前抛 NotImplementedError）

设计要点：
- 客户端用 `functools.lru_cache` 做惰性单例（满足 AGENTS Hard Rule 10，且 mock 友好：
  测试只需 patch `_get_client`，缓存自然失效）。
- 模块顶层不调用 `load_dotenv`、不建 client，全部延迟到首次调用，便于测试。
- 为了 DeepSeek prompt caching，`chat_structured` 的 system 参数应由调用方传入
  *模块级常量*（不含 f-string 变量），见 AGENTS.md 的 LLM Call Pattern。
- 绝不把 prompt / response 内容写进日志（含异常 traceback，因 loguru diagnose
  默认会展开局部变量）。
"""

import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
# 仓库根目录下的 config/secrets.env（与 tests/test_p0_05_apis.py 的约定一致）。
SECRETS_PATH = Path(__file__).resolve().parents[2] / "config" / "secrets.env"


class LLMError(Exception):
    """所有 LLM 层错误的基类，调用方可统一捕获。"""


class LLMConfigError(LLMError):
    """配置缺失或非法（如缺少 API key）。"""


class LLMResponseError(LLMError):
    """模型返回了空内容或结构异常。"""


class LLMJSONError(LLMResponseError):
    """`chat_structured` 期望 JSON 对象，但模型返回了无法解析的内容。"""


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    """惰性构建并缓存 DeepSeek 客户端（OpenAI 兼容）。

    在首次真实调用时才读取 secrets，避免导入期副作用。测试通过 patch 本函数
    （`src.ielts.llm._get_client`）即可绕过缓存与真实网络。

    Returns:
        已配置 api_key / base_url 的 OpenAI 客户端单例。

    Raises:
        LLMConfigError: 当 DEEPSEEK_API_KEY 未设置时。
    """
    # load_dotenv 默认不覆盖已存在的环境变量，因此测试可用 monkeypatch 注入。
    load_dotenv(SECRETS_PATH)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise LLMConfigError("DEEPSEEK_API_KEY 未设置；请检查 config/secrets.env 或环境变量。")
    base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


def _complete(
    messages: list[dict],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    response_format: dict | None = None,
) -> str:
    """底层补全调用，被 chat / chat_structured 共用。

    注意：不在此处捕获 openai 的异常——openai 库自带重试与清晰的异常类型，
    且捕获后用 loguru 记录 traceback 有泄露 prompt 的风险。让其自然向上抛出。
    """
    if not messages:
        raise LLMError("messages 不能为空。")

    logger.debug(
        "LLM call model={} max_tokens={} n_messages={} json={}",
        model,
        max_tokens,
        len(messages),
        response_format is not None,
    )

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format

    response = _get_client().chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    if not content:
        raise LLMResponseError("模型返回了空内容。")
    return content


def chat(
    messages: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    timeout: float = 60.0,
) -> str:
    """发起一次纯文本对话补全。

    Args:
        messages: OpenAI 风格的消息列表，如
            `[{"role": "user", "content": "..."}]`。
        model: 模型名，默认 deepseek-chat。
        temperature: 采样温度。
        max_tokens: 最大生成 token 数。
        timeout: 单次请求超时（秒）。

    Returns:
        模型回复的文本内容（非空）。

    Raises:
        LLMResponseError: 模型返回空内容。
        LLMConfigError: API key 未配置。
    """
    return _complete(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def chat_structured(
    system: str,
    user: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    timeout: float = 60.0,
) -> dict:
    """发起一次强制 JSON 输出的对话补全。

    使用 DeepSeek 的 `response_format={"type": "json_object"}`。按 DeepSeek 要求，
    调用方的 system 或 user 文本中必须出现 "JSON" 字样，否则模型可能拒绝。
    为利用 prompt caching，system 建议是模块级常量。

    Args:
        system: 系统提示词（建议模块级常量，缓存友好）。
        user: 用户输入（可含动态内容）。
        model: 模型名，默认 deepseek-chat。
        temperature: 采样温度。
        max_tokens: 最大生成 token 数。
        timeout: 单次请求超时（秒）。

    Returns:
        解析后的 JSON 对象（dict）。

    Raises:
        LLMJSONError: 返回内容不是合法 JSON，或不是 JSON 对象。
        LLMResponseError: 模型返回空内容。
        LLMConfigError: API key 未配置。
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    content = _complete(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        response_format={"type": "json_object"},
    )

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        # 只记录异常类型，不记录 content（避免泄露 response）。
        logger.error("chat_structured 收到非 JSON 响应：{}", exc.__class__.__name__)
        raise LLMJSONError("模型未返回合法 JSON。") from exc

    if not isinstance(data, dict):
        raise LLMJSONError(f"期望 JSON 对象，实际得到 {type(data).__name__}。")
    return data


def embed(texts: list[str]) -> list[list[float]]:
    """文本嵌入（占位）。

    嵌入功能计划在 P4 用 SiliconFlow（见 secrets.env 的 SILICONFLOW_*）实现，
    目前未实现。

    Args:
        texts: 待嵌入的文本列表。

    Raises:
        NotImplementedError: 永远——该功能尚未实现。
    """
    raise NotImplementedError("embed 将在 P4 用 SiliconFlow 实现；当前阶段未启用。")
