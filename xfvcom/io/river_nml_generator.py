# SPDX-License-Identifier: MIT
"""
Generator for &NML_RIVER namelist blocks.
"""

from __future__ import annotations

import csv
import json
from typing import Any

import yaml
from jinja2 import BaseLoader, Environment, select_autoescape

from .base_generator import BaseGenerator


class RiverNmlGenerator(BaseGenerator):
    """Generate &NML_RIVER blocks from CSV / YAML / JSON files."""

    template_name = "river_namelist.j2"

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def load(self) -> None:  # noqa: D401
        """Detect the file format from suffix and call the appropriate loader."""
        suffix = self.source.suffix.lower()
        if suffix == ".csv":
            self._load_csv()
        elif suffix in {".yaml", ".yml"}:
            self._load_yaml()
        elif suffix == ".json":
            self._load_json()
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def validate(self) -> None:  # noqa: D401
        """Raise if *self.rows* is empty or missing required keys."""
        if not self.rows:
            raise ValueError("No river entries found.")
        required = {"RIVER_NAME", "RIVER_GRID_LOCATION", "RIVER_VERTICAL_DISTRIBUTION"}
        for i, row in enumerate(self.rows, start=1):
            if not required.issubset(row):
                raise ValueError(f"Row {i} missing required keys: {row}")

    def render(self) -> str:
        """Render the Jinja2 template with populated rows."""
        tmpl_text = self._get_template(self.template_name)
        env = Environment(loader=BaseLoader(), autoescape=select_autoescape())
        tmpl = env.from_string(tmpl_text)

        blocks: list[str] = []
        for row in self.rows:
            blocks.append(
                tmpl.render(
                    name=row["RIVER_NAME"],
                    river_file=self.river_file,
                    grid=int(row["RIVER_GRID_LOCATION"]),
                    vertical=row["RIVER_VERTICAL_DISTRIBUTION"],
                )
            )
        return "\n".join(blocks)

    # --------------------------------------------------------------------- #
    # Individual loaders                                                    #
    # --------------------------------------------------------------------- #
    def _load_csv(self) -> None:
        """Load river definition from a two-header CSV file."""
        with self.source.open(newline="", encoding="utf-8") as fp:
            reader = csv.reader(fp)

            # 1) global river file
            first_row = next(reader, None)
            if not first_row or first_row[0] != "RIVER_FILE":
                raise ValueError("First line must start with 'RIVER_FILE,<filename>'")
            self.river_file = first_row[1].strip()

            # 2) required header
            expected = (
                "RIVER_NAME",
                "RIVER_GRID_LOCATION",
                "RIVER_VERTICAL_DISTRIBUTION",
            )
            header = next(reader, None)
            if header is None or tuple(header[:3]) != expected:
                raise ValueError(f"Second line must be header {expected}")

            # 3) data rows
            self.rows = []
            for row in reader:
                if not row or all(c.strip() == "" for c in row):
                    continue
                name, grid, vertical = (c.strip() for c in row[:3])
                self.rows.append(
                    {
                        "RIVER_NAME": name,
                        "RIVER_GRID_LOCATION": grid,
                        "RIVER_VERTICAL_DISTRIBUTION": vertical,
                    }
                )

    def _load_yaml(self) -> None:
        """Load river definition from a YAML mapping."""
        data: dict[str, Any] = yaml.safe_load(self.source.read_text(encoding="utf-8"))
        self._parse_mapping(data)

    def _load_json(self) -> None:
        """Load river definition from a JSON mapping."""
        data: dict[str, Any] = json.loads(self.source.read_text(encoding="utf-8"))
        self._parse_mapping(data)

    # --------------------------------------------------------------------- #
    # Shared helper                                                         #
    # --------------------------------------------------------------------- #
    def _parse_mapping(self, data: dict[str, Any]) -> None:
        """Populate internal fields from a mapping object."""
        try:
            self.river_file = data["river_file"]
            rivers = data["rivers"]
        except KeyError as exc:
            raise ValueError(f"Missing key: {exc}") from exc

        self.rows = []
        for r in rivers:
            self.rows.append(
                {
                    "RIVER_NAME": r["name"],
                    "RIVER_GRID_LOCATION": str(r["grid"]),
                    "RIVER_VERTICAL_DISTRIBUTION": r.get("vertical", "uniform"),
                }
            )
