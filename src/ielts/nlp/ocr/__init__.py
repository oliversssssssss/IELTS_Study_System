"""Tesseract OCR wrapper for IELTS reading image intake."""

import re
from pathlib import Path

import pytesseract
from loguru import logger
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_LANG = "eng"
TESSERACT_OEM = 3
PSM_MIN = 0
PSM_MAX = 13


class OCRError(Exception):
    """OCR failed because the input or Tesseract execution was invalid."""


def _validate_options(lang: str, psm: int) -> None:
    """Validate OCR options supported by this project."""
    if lang != SUPPORTED_LANG:
        raise OCRError(f"Only English OCR is supported: lang={lang!r}.")
    if not isinstance(psm, int) or isinstance(psm, bool):
        raise OCRError(f"psm must be an integer between {PSM_MIN} and {PSM_MAX}.")
    if not PSM_MIN <= psm <= PSM_MAX:
        raise OCRError(f"psm must be between {PSM_MIN} and {PSM_MAX}: {psm}.")


def _resolve_image_path(path: str | Path) -> Path:
    """Resolve and validate an OCR input path."""
    image_path = Path(path).expanduser()
    if not image_path.exists():
        raise FileNotFoundError(f"OCR image not found: {image_path}")
    if not image_path.is_file():
        raise OCRError(f"OCR path is not a file: {image_path}")
    return image_path


def _tesseract_config(psm: int) -> str:
    """Build the Tesseract config string used for IELTS reading images."""
    return f"--oem {TESSERACT_OEM} --psm {psm}"


def _prepare_image(image: Image.Image) -> Image.Image:
    """Convert image to grayscale and improve contrast before OCR."""
    return ImageOps.autocontrast(image.convert("L"))


def _clean_ocr_text(raw: str) -> str:
    """Normalize common OCR line wrapping artifacts while preserving paragraphs."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)", "", text)

    paragraphs = re.split(r"\n[ \t]*\n+", text.strip())
    cleaned_paragraphs = []
    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
        cleaned = re.sub(r"[ \t]+", " ", " ".join(lines)).strip()
        if cleaned:
            cleaned_paragraphs.append(cleaned)

    return "\n\n".join(cleaned_paragraphs)


def ocr_image(
    path: str | Path,
    *,
    lang: str = SUPPORTED_LANG,
    psm: int = 6,
) -> str:
    """Run OCR on a single image and return cleaned plain text.

    Args:
        path: Image file path.
        lang: Tesseract language code. Only English (`eng`) is supported.
        psm: Tesseract page segmentation mode. The default `6` fits a single
            text block, which is suitable for IELTS reading passage photos.

    Returns:
        Cleaned OCR text.

    Raises:
        FileNotFoundError: The image path does not exist.
        OCRError: The image cannot be read, options are unsupported, or
            Tesseract fails.
    """
    image_path = _resolve_image_path(path)
    _validate_options(lang, psm)

    try:
        with Image.open(image_path) as image:
            prepared_image = _prepare_image(image)
            raw_text = pytesseract.image_to_string(
                prepared_image,
                lang=lang,
                config=_tesseract_config(psm),
            )
    except FileNotFoundError:
        raise
    except (pytesseract.TesseractError, pytesseract.TesseractNotFoundError) as exc:
        logger.exception("Tesseract OCR failed for image: {}", image_path)
        raise OCRError(f"Tesseract OCR failed for image: {image_path}") from exc
    except (UnidentifiedImageError, OSError) as exc:
        logger.exception("OCR image could not be read: {}", image_path)
        raise OCRError(f"OCR image could not be read: {image_path}") from exc

    return _clean_ocr_text(raw_text)


def ocr_images(
    paths: list[str | Path],
    *,
    lang: str = SUPPORTED_LANG,
    psm: int = 6,
) -> str:
    """Run OCR on multiple images and join them with paragraph breaks.

    Args:
        paths: Image file paths, in reading order.
        lang: Tesseract language code. Only English (`eng`) is supported.
        psm: Tesseract page segmentation mode.

    Returns:
        Cleaned OCR text from all images, separated by blank lines.

    Raises:
        FileNotFoundError: Any image path does not exist.
        OCRError: Any image cannot be read, options are unsupported, or
            Tesseract fails.
    """
    texts = [ocr_image(path, lang=lang, psm=psm) for path in paths]
    return "\n\n".join(text for text in texts if text)
