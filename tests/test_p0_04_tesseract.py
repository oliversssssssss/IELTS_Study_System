"""Smoke test for Tesseract OCR installation."""

import subprocess

import pytesseract
from PIL import Image, ImageDraw, ImageFont


def test_tesseract_binary_available() -> None:
    """Tesseract system binary must be installed."""
    result = subprocess.run(
        ["tesseract", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "tesseract" in result.stdout.lower() or "tesseract" in result.stderr.lower()


def test_tesseract_english_lang_available() -> None:
    """English language pack must be installed."""
    langs = pytesseract.get_languages()
    assert "eng" in langs


def test_pytesseract_can_ocr_text() -> None:
    """Generate an image with text, OCR it, verify result."""
    img = Image.new("RGB", (500, 120), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=36)
    draw.text((20, 35), "Hello IELTS World", fill="black", font=font)

    text = pytesseract.image_to_string(img, lang="eng")

    text_lower = text.lower()
    assert "hello" in text_lower
    assert "ielts" in text_lower or "world" in text_lower
