"""Write evaluation reports without leaking local absolute paths."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from contextle_eval.metrics import EvaluationResult


class ReportWriteError(RuntimeError):
    """Raised when report files cannot be created."""


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_reports(
    output_dir: Path,
    result: EvaluationResult,
    run_config: dict[str, object],
) -> None:
    """Create the documented five-file report bundle in a new empty directory."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not output_dir.is_dir():
            raise OSError("path exists but is not a directory")
        if any(output_dir.iterdir()):
            raise ReportWriteError(
                f"Output directory is not empty: {output_dir}. "
                "Use a new versioned directory so previous evaluation results are preserved."
            )
        _write_json(output_dir / "summary.json", result.summary)
        _write_csv(output_dir / "detailed_results.csv", result.detailed_rows)
        _write_csv(output_dir / "per_answer_metrics.csv", result.per_answer_rows)
        _write_csv(output_dir / "top_k_results.csv", result.top_k_rows)
        _write_json(output_dir / "run_config.json", run_config)
    except OSError as exc:
        raise ReportWriteError(
            f"Could not create evaluation reports in {output_dir}: {exc}. "
            "Check that the parent exists and is writable."
        ) from exc
