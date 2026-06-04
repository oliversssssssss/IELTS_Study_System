"""Unit tests for the Tesseract OCR wrapper.

The tests patch `pytesseract.image_to_string` so they never call the real OCR
binary. Pillow is only used to create tiny local image files for path handling.
"""

from pathlib import Path

import pytesseract
import pytest
from PIL import Image

from src.ielts.nlp.ocr import OCRError, _clean_ocr_text, ocr_image, ocr_images


def _make_image(path: Path) -> Path:
    """Create a tiny valid image file for OCR wrapper tests."""
    Image.new("RGB", (80, 40), color="white").save(path)
    return path


def test_ocr_image_calls_tesseract_with_default_options_and_cleans_text(
    tmp_path: Path,
    mocker,
) -> None:
    image_path = _make_image(tmp_path / "page.png")
    image_to_string = mocker.patch(
        "src.ielts.nlp.ocr.pytesseract.image_to_string",
        return_value="The cat\nsits on the ele-\nphant.\n\nSecond   paragraph.\n",
    )

    result = ocr_image(image_path)

    assert result == "The cat sits on the elephant.\n\nSecond paragraph."
    image_to_string.assert_called_once()
    ocr_image_arg = image_to_string.call_args.args[0]
    assert isinstance(ocr_image_arg, Image.Image)
    assert ocr_image_arg.mode == "L"
    assert image_to_string.call_args.kwargs == {
        "lang": "eng",
        "config": "--oem 3 --psm 6",
    }


def test_ocr_image_accepts_string_path_and_custom_psm(tmp_path: Path, mocker) -> None:
    image_path = _make_image(tmp_path / "page.png")
    image_to_string = mocker.patch(
        "src.ielts.nlp.ocr.pytesseract.image_to_string",
        return_value="Sparse text",
    )

    result = ocr_image(str(image_path), psm=11)

    assert result == "Sparse text"
    assert image_to_string.call_args.kwargs["config"] == "--oem 3 --psm 11"


def test_ocr_images_concatenates_multiple_inputs_with_blank_line(tmp_path: Path, mocker) -> None:
    first_path = _make_image(tmp_path / "first.png")
    second_path = _make_image(tmp_path / "second.png")
    mocker.patch(
        "src.ielts.nlp.ocr.pytesseract.image_to_string",
        side_effect=["First page", "Second\npage"],
    )

    result = ocr_images([first_path, second_path])

    assert result == "First page\n\nSecond page"


def test_ocr_images_skips_empty_ocr_results_when_joining(tmp_path: Path, mocker) -> None:
    first_path = _make_image(tmp_path / "first.png")
    second_path = _make_image(tmp_path / "second.png")
    mocker.patch(
        "src.ielts.nlp.ocr.pytesseract.image_to_string",
        side_effect=["", "Second page"],
    )

    result = ocr_images([first_path, second_path])

    assert result == "Second page"


def test_ocr_images_empty_list_returns_empty_text(mocker) -> None:
    image_to_string = mocker.patch("src.ielts.nlp.ocr.pytesseract.image_to_string")

    result = ocr_images([])

    assert result == ""
    image_to_string.assert_not_called()


def test_ocr_image_missing_file_raises_file_not_found(tmp_path: Path, mocker) -> None:
    image_to_string = mocker.patch("src.ielts.nlp.ocr.pytesseract.image_to_string")

    with pytest.raises(FileNotFoundError):
        ocr_image(tmp_path / "missing.png")

    image_to_string.assert_not_called()


def test_ocr_image_directory_raises_ocr_error(tmp_path: Path, mocker) -> None:
    image_to_string = mocker.patch("src.ielts.nlp.ocr.pytesseract.image_to_string")

    with pytest.raises(OCRError):
        ocr_image(tmp_path)

    image_to_string.assert_not_called()


def test_ocr_image_unreadable_file_raises_ocr_error(tmp_path: Path, mocker) -> None:
    image_path = tmp_path / "not-image.txt"
    image_path.write_text("not an image", encoding="utf-8")
    image_to_string = mocker.patch("src.ielts.nlp.ocr.pytesseract.image_to_string")

    with pytest.raises(OCRError):
        ocr_image(image_path)

    image_to_string.assert_not_called()


def test_ocr_image_wraps_tesseract_error(tmp_path: Path, mocker) -> None:
    image_path = _make_image(tmp_path / "page.png")
    mocker.patch(
        "src.ielts.nlp.ocr.pytesseract.image_to_string",
        side_effect=pytesseract.TesseractError(1, "bad image"),
    )

    with pytest.raises(OCRError):
        ocr_image(image_path)


def test_ocr_image_rejects_non_english_language(tmp_path: Path, mocker) -> None:
    image_path = _make_image(tmp_path / "page.png")
    image_to_string = mocker.patch("src.ielts.nlp.ocr.pytesseract.image_to_string")

    with pytest.raises(OCRError):
        ocr_image(image_path, lang="chi_sim")

    image_to_string.assert_not_called()


def test_ocr_image_rejects_invalid_psm(tmp_path: Path, mocker) -> None:
    image_path = _make_image(tmp_path / "page.png")
    image_to_string = mocker.patch("src.ielts.nlp.ocr.pytesseract.image_to_string")

    with pytest.raises(OCRError):
        ocr_image(image_path, psm=14)

    image_to_string.assert_not_called()


def test_ocr_image_rejects_bool_psm(tmp_path: Path, mocker) -> None:
    image_path = _make_image(tmp_path / "page.png")
    image_to_string = mocker.patch("src.ielts.nlp.ocr.pytesseract.image_to_string")

    with pytest.raises(OCRError):
        ocr_image(image_path, psm=True)

    image_to_string.assert_not_called()


def test_clean_ocr_text_preserves_paragraph_breaks_and_removes_wraps() -> None:
    raw = " First line\ncontinues here \n\n New para\ncontinues."

    result = _clean_ocr_text(raw)

    assert result == "First line continues here\n\nNew para continues."
