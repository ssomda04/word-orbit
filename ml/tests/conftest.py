"""Make the ML source package importable without installing it."""

from __future__ import annotations

import sys
from pathlib import Path

ML_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ML_SRC))
