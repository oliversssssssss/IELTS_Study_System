"""Unit tests for Bark push wrapper — 全部 mock，不调用真实 Bark API。"""

import requests

from src.ielts.tools import push
from src.ielts.tools.push import push_bark


def _fake_response(
    mocker,
    *,
    status_code: int = 200,
    payload=None,
    json_error: Exception | None = None,
):
    """构造最小 requests.Response 替身。"""
    response = mocker.Mock()
    response.status_code = status_code
    if json_error is not None:
        response.json.side_effect = json_error
    else:
        response.json.return_value = (
            {"code": 200, "message": "success"} if payload is None else payload
        )
    return response


def test_push_bark_success_minimal_payload(mocker, monkeypatch) -> None:
    mocker.patch("src.ielts.tools.push.load_dotenv")
    monkeypatch.setenv("BARK_URL", "https://bark.example/key")
    response = _fake_response(mocker)
    post = mocker.patch("src.ielts.tools.push.requests.post", return_value=response)

    result = push_bark("Title", "Body")

    assert result is True
    post.assert_called_once_with(
        "https://bark.example/key",
        json={"title": "Title", "body": "Body"},
        timeout=push.BARK_TIMEOUT_SECONDS,
    )


def test_push_bark_includes_optional_payload_fields(mocker, monkeypatch) -> None:
    mocker.patch("src.ielts.tools.push.load_dotenv")
    monkeypatch.setenv("BARK_URL", "https://bark.example/key")
    response = _fake_response(mocker)
    post = mocker.patch("src.ielts.tools.push.requests.post", return_value=response)

    result = push_bark(
        "Title",
        "Body",
        group="ielts",
        url="obsidian://open?vault=ielts",
        sound="default",
        level="active",
    )

    assert result is True
    payload = post.call_args.kwargs["json"]
    assert payload == {
        "title": "Title",
        "body": "Body",
        "group": "ielts",
        "url": "obsidian://open?vault=ielts",
        "sound": "default",
        "level": "active",
    }


def test_push_bark_missing_url_returns_false(mocker, monkeypatch) -> None:
    mocker.patch("src.ielts.tools.push.load_dotenv")
    monkeypatch.delenv("BARK_URL", raising=False)
    post = mocker.patch("src.ielts.tools.push.requests.post")
    log_error = mocker.patch("src.ielts.tools.push.logger.error")

    result = push_bark("Title", "Body")

    assert result is False
    post.assert_not_called()
    log_error.assert_called_once()


def test_push_bark_request_exception_returns_false(mocker, monkeypatch) -> None:
    mocker.patch("src.ielts.tools.push.load_dotenv")
    monkeypatch.setenv("BARK_URL", "https://bark.example/key")
    mocker.patch(
        "src.ielts.tools.push.requests.post",
        side_effect=requests.Timeout("timeout"),
    )
    log_error = mocker.patch("src.ielts.tools.push.logger.error")

    result = push_bark("Title", "Body")

    assert result is False
    assert log_error.call_args.args[0] == "Bark push request failed: {}"


def test_push_bark_http_error_returns_false(mocker, monkeypatch) -> None:
    mocker.patch("src.ielts.tools.push.load_dotenv")
    monkeypatch.setenv("BARK_URL", "https://bark.example/key")
    response = _fake_response(mocker, status_code=500)
    mocker.patch("src.ielts.tools.push.requests.post", return_value=response)
    log_error = mocker.patch("src.ielts.tools.push.logger.error")

    result = push_bark("Title", "Body")

    assert result is False
    response.json.assert_not_called()
    assert log_error.call_args.args[0] == "Bark push HTTP failed: status_code={}"


def test_push_bark_invalid_json_returns_false(mocker, monkeypatch) -> None:
    mocker.patch("src.ielts.tools.push.load_dotenv")
    monkeypatch.setenv("BARK_URL", "https://bark.example/key")
    response = _fake_response(mocker, json_error=ValueError("not json"))
    mocker.patch("src.ielts.tools.push.requests.post", return_value=response)
    log_error = mocker.patch("src.ielts.tools.push.logger.error")

    result = push_bark("Title", "Body")

    assert result is False
    assert log_error.call_args.args[0] == "Bark push returned non-JSON response: {}"


def test_push_bark_bark_failure_returns_false(mocker, monkeypatch) -> None:
    mocker.patch("src.ielts.tools.push.load_dotenv")
    monkeypatch.setenv("BARK_URL", "https://bark.example/key")
    response = _fake_response(mocker, payload={"code": 400, "message": "bad request"})
    mocker.patch("src.ielts.tools.push.requests.post", return_value=response)
    log_error = mocker.patch("src.ielts.tools.push.logger.error")

    result = push_bark("Title", "Body")

    assert result is False
    assert log_error.call_args.args[0] == "Bark push failed: code={} message={}"


def test_push_bark_non_object_json_returns_false(mocker, monkeypatch) -> None:
    mocker.patch("src.ielts.tools.push.load_dotenv")
    monkeypatch.setenv("BARK_URL", "https://bark.example/key")
    response = _fake_response(mocker, payload=["unexpected"])
    mocker.patch("src.ielts.tools.push.requests.post", return_value=response)
    log_error = mocker.patch("src.ielts.tools.push.logger.error")

    result = push_bark("Title", "Body")

    assert result is False
    assert log_error.call_args.args[0] == "Bark push failed: non-object response type={}"


def test_get_bark_url_loads_dotenv_once(mocker, monkeypatch) -> None:
    load_dotenv = mocker.patch("src.ielts.tools.push.load_dotenv")
    monkeypatch.setenv("BARK_URL", "https://bark.example/key")

    assert push._get_bark_url() == "https://bark.example/key"
    assert push._get_bark_url() == "https://bark.example/key"
    load_dotenv.assert_called_once_with(push.SECRETS_PATH)


def setup_function() -> None:
    push._get_bark_url.cache_clear()


def teardown_function() -> None:
    push._get_bark_url.cache_clear()
