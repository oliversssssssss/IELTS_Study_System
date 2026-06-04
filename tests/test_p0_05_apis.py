"""Smoke test for API key configuration.

Verifies that API keys in config/secrets.env can authenticate with
their respective services. Keys are NEVER printed or logged.
"""

import os
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests
from dotenv import load_dotenv

SECRETS_PATH = Path(__file__).parent.parent / "config" / "secrets.env"
load_dotenv(SECRETS_PATH)


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise AssertionError(f"{name} not set")
    return value


def test_secrets_file_exists() -> None:
    """The secrets file must exist."""
    assert SECRETS_PATH.exists(), f"Missing {SECRETS_PATH}"


def test_secrets_file_permissions() -> None:
    """Secrets file should have restrictive permissions on real Linux fs.

    Skipped on Windows mount (NTFS) where Unix permissions are not enforced.
    """
    resolved = SECRETS_PATH.resolve()
    if str(resolved).startswith("/mnt/"):
        pytest.skip("Cannot enforce file permissions on Windows mount")

    mode = SECRETS_PATH.stat().st_mode & 0o777
    assert mode <= 0o600, f"Insecure permissions: {oct(mode)}"


def test_deepseek_key_present() -> None:
    """DEEPSEEK_API_KEY must be set and look valid."""
    key = _get_required_env("DEEPSEEK_API_KEY")
    if not key.startswith("sk-"):
        raise AssertionError("DEEPSEEK_API_KEY format unexpected")
    if len(key) <= 20:
        raise AssertionError("DEEPSEEK_API_KEY seems too short")


def test_deepseek_api_reachable() -> None:
    """Verify DeepSeek API authenticates and responds with a real call."""
    from openai import OpenAI, OpenAIError

    key = _get_required_env("DEEPSEEK_API_KEY")
    client = OpenAI(
        api_key=key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Reply with the word OK only."}],
            max_tokens=5,
            temperature=0,
        )
    except (OpenAIError, UnicodeEncodeError) as exc:
        error_type = exc.__class__.__name__
        raise AssertionError(f"DeepSeek API call failed: {error_type}") from None

    assert response.choices[0].message.content


def test_bark_url_present() -> None:
    """BARK_URL must be set and follow the expected format."""
    url = _get_required_env("BARK_URL")
    if not url.startswith("https://api.day.app/"):
        raise AssertionError("BARK_URL format unexpected")
    if not url.endswith("/"):
        raise AssertionError("BARK_URL should end with /")


def test_bark_reachable() -> None:
    """Verify Bark endpoint responds without sending a push."""
    url = _get_required_env("BARK_URL")

    parsed = urlparse(url)
    ping_url = f"{parsed.scheme}://{parsed.netloc}/ping"

    try:
        response = requests.get(ping_url, timeout=10)
    except requests.RequestException as exc:
        error_type = exc.__class__.__name__
        raise AssertionError(f"Bark server unreachable: {error_type}") from None

    assert response.status_code in (200, 404), (
        f"Bark server unreachable: status {response.status_code}"
    )


@pytest.mark.skipif(
    not os.getenv("SILICONFLOW_API_KEY"),
    reason="SILICONFLOW_API_KEY not set (optional for P0)",
)
def test_siliconflow_key_present() -> None:
    """SILICONFLOW_API_KEY format check (optional, only if set)."""
    key = _get_required_env("SILICONFLOW_API_KEY")
    if not key.startswith("sk-"):
        raise AssertionError("SILICONFLOW_API_KEY format unexpected")
