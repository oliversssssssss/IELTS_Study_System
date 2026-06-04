"""Bark push notification wrapper."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from loguru import logger

BARK_TIMEOUT_SECONDS = 10
BARK_SUCCESS_CODE = 200
HTTP_SUCCESS_MIN = 200
HTTP_SUCCESS_MAX = 300
# 仓库根目录下的 config/secrets.env；只用于加载环境变量，日志中绝不输出其内容。
SECRETS_PATH = Path(__file__).resolve().parents[4] / "config" / "secrets.env"


@lru_cache(maxsize=1)
def _get_bark_url() -> str | None:
    """惰性读取 Bark 推送 URL。

    `load_dotenv` 默认不覆盖已存在环境变量，测试可用 monkeypatch 注入 BARK_URL。
    """
    load_dotenv(SECRETS_PATH)
    return os.getenv("BARK_URL")


def _build_payload(
    title: str,
    body: str,
    *,
    group: str | None,
    url: str | None,
    sound: str | None,
    level: str | None,
) -> dict[str, str]:
    """构建 Bark JSON payload，仅包含用户实际传入的可选字段。"""
    payload = {
        "title": title,
        "body": body,
    }
    optional_payload = {
        "group": group,
        "url": url,
        "sound": sound,
        "level": level,
    }
    payload.update({key: value for key, value in optional_payload.items() if value is not None})
    return payload


def _is_success_response(data: Any) -> bool:
    """判断 Bark JSON 响应是否表示推送成功。"""
    return isinstance(data, dict) and data.get("code") == BARK_SUCCESS_CODE


def push_bark(
    title: str,
    body: str,
    *,
    group: str | None = None,
    url: str | None = None,
    sound: str | None = None,
    level: str | None = None,
) -> bool:
    """发送 Bark 手机推送。

    Args:
        title: 通知标题。
        body: 通知内容。
        group: Bark 通知分组。
        url: 点击通知后跳转的 URL。
        sound: Bark 自定义铃声名。
        level: Bark 通知级别，如 active / timeSensitive / passive。

    Returns:
        True 表示 Bark 返回成功；False 表示配置缺失、请求失败或 Bark 返回失败。
    """
    bark_url = _get_bark_url()
    if not bark_url:
        logger.error("Bark push skipped: BARK_URL is not configured.")
        return False

    payload = _build_payload(
        title,
        body,
        group=group,
        url=url,
        sound=sound,
        level=level,
    )

    try:
        response = requests.post(bark_url, json=payload, timeout=BARK_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        logger.error("Bark push request failed: {}", exc.__class__.__name__)
        return False

    if not HTTP_SUCCESS_MIN <= response.status_code < HTTP_SUCCESS_MAX:
        logger.error("Bark push HTTP failed: status_code={}", response.status_code)
        return False

    try:
        data = response.json()
    except ValueError as exc:
        logger.error("Bark push returned non-JSON response: {}", exc.__class__.__name__)
        return False

    if _is_success_response(data):
        logger.debug("Bark push sent successfully.")
        return True

    if isinstance(data, dict):
        logger.error(
            "Bark push failed: code={} message={}",
            data.get("code"),
            data.get("message"),
        )
    else:
        logger.error("Bark push failed: non-object response type={}", type(data).__name__)
    return False
