"""Import guard for the P0 smoke module."""

import importlib
from types import ModuleType


def test_smoke_p0_imports_successfully() -> None:
    """The smoke module must import without external calls."""
    module = importlib.import_module("src.ielts.smoke_p0")

    assert isinstance(module, ModuleType)
