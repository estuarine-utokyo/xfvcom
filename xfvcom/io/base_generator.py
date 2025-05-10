# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""
Common utilities for FVCOM input file generators.
"""
from __future__ import annotations

import importlib.resources
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseGenerator(ABC):
    """Abstract base class for all FVCOM input generators."""

    # ↓ 継承側で上書きする
    template_name: str

    def __init__(self, source: Path) -> None:
        self.source = source.expanduser().resolve()
        self.data: Any = None  # Loaded representation

    # ---------- High-level public API ---------- #

    def generate(self) -> str | bytes:
        """Return rendered text (or binary) ready to be written to file."""
        self.load()
        self.validate()
        return self.render()

    def write(self, dest: Path | None = None) -> Path:
        """Generate file and write to *dest* (default: <source stem>.out)."""
        if dest is None:
            dest = self.source.with_suffix(".out")

        output = self.generate()
        mode = "wb" if isinstance(output, bytes) else "w"
        encoding: str | None = None if mode == "wb" else "utf-8"
        with dest.open(mode, encoding=encoding) as fp:  # type: ignore[arg-type]
            fp.write(output)
        return dest

    # ---------- Steps overridable by subclasses ---------- #

    @abstractmethod
    def load(self) -> None:  # noqa: D401
        """Read *self.source* into *self.data*."""
        ...

    @abstractmethod
    def validate(self) -> None:  # noqa: D401
        """Raise if *self.data* is invalid."""
        ...

    @abstractmethod
    def render(self) -> str | bytes:  # noqa: D401
        """Return final content from *self.data*."""
        ...

    # ---------- Helper ---------- #

    @staticmethod
    def _get_template(name: str) -> str:
        """Load text template from package resources."""
        pkg = __name__.rsplit(".", 1)[0] + ".templates"
        return importlib.resources.files(pkg).joinpath(name).read_text(encoding="utf-8")
