"""Read-only verification of completed raw genre-frequency production outputs."""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

Genre = Literal["newspaper", "dialogue", "online"]
GENRES: tuple[Genre, ...] = ("newspaper", "dialogue", "online")
REPORT_SCHEMA_VERSION = "1.0"
FREQUENCY_FIELDS = ("canonical_form", "count")
CANDIDATE_REQUIRED_FIELDS = frozenset(
    {
        "source",
        "source_text_id",
        "canonical_form",
        "frequency_assignment",
    }
)
CONSERVATION_FLAGS = (
    "status_conserved",
    "roles_conserved",
    "base_equals_candidate_count",
)
FIXED_REPORT_MAPPINGS = (
    "normalization_status",
    "lexical_roles",
    "frequency",
    "major_pos_and_suffix_distribution",
    "placeholder_audit",
    "markup_audit",
    "parser",
    "count_conservation",
    "performance",
    "outputs",
)


@dataclass(slots=True)
class GenreVerification:
    """One genre's human- and machine-readable verification outcome."""

    genre: Genre
    passed: bool = True
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    report_schema_signature: dict[str, list[str]] | None = None

    def fail(self, message: str) -> None:
        self.passed = False
        self.failures.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "genre": self.genre,
            "passed": self.passed,
            "failures": self.failures,
            "warnings": self.warnings,
            "summary": self.summary,
        }


@dataclass(slots=True)
class VerificationResult:
    """All requested genre checks plus their cross-report consistency result."""

    genres: list[GenreVerification]
    schema_consistent: bool
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.schema_consistent and not self.failures and all(
            genre.passed for genre in self.genres
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "schema_consistent": self.schema_consistent,
            "failures": self.failures,
            "genres": [genre.as_dict() for genre in self.genres],
        }


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _mapping(
    report: Mapping[str, Any], key: str, result: GenreVerification
) -> Mapping[str, Any] | None:
    value = report.get(key)
    if not isinstance(value, dict):
        result.fail(f"report.{key} must be an object")
        return None
    return value


def _nonnegative_mapping(
    report: Mapping[str, Any], key: str, result: GenreVerification
) -> Mapping[str, int] | None:
    value = _mapping(report, key, result)
    if value is None:
        return None
    invalid = sorted(name for name, count in value.items() if not _is_nonnegative_int(count))
    if invalid:
        result.fail(f"report.{key} has invalid non-negative integer fields: {invalid}")
        return None
    return value  # type: ignore[return-value]


def _file_sizes(paths: Mapping[str, Path]) -> dict[str, int | None]:
    return {
        name: path.stat().st_size if path.is_file() else None
        for name, path in paths.items()
    }


def _verify_frequency_csv(
    path: Path, result: GenreVerification
) -> tuple[int, int] | None:
    total = 0
    row_count = 0
    seen: set[str] = set()
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(FREQUENCY_FIELDS):
                result.fail(
                    f"frequency CSV header must be {list(FREQUENCY_FIELDS)}, "
                    f"got {reader.fieldnames}"
                )
                return None
            for line_number, row in enumerate(reader, start=2):
                canonical = row["canonical_form"]
                if not canonical:
                    result.fail(f"frequency CSV row {line_number} has a blank canonical form")
                if canonical in seen:
                    result.fail(
                        f"frequency CSV has duplicate canonical form at row {line_number}"
                    )
                seen.add(canonical)
                raw_count = row["count"]
                try:
                    count = int(raw_count)
                except ValueError:
                    result.fail(
                        f"frequency CSV row {line_number} has invalid count {raw_count!r}"
                    )
                    continue
                if count < 0 or str(count) != raw_count:
                    result.fail(
                        f"frequency CSV row {line_number} count must be a canonical "
                        "non-negative integer"
                    )
                    continue
                total += count
                row_count += 1
    except (OSError, UnicodeError, csv.Error) as exc:
        result.fail(f"could not read frequency CSV: {exc}")
        return None
    return total, row_count


def _verify_candidate_gzip(
    path: Path, result: GenreVerification
) -> tuple[list[str], int] | None:
    """Stream the complete compressed CSV and return its header and data-row count."""
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader, None)
            if header is None:
                result.fail("candidate gzip is empty")
                return None
            missing = sorted(CANDIDATE_REQUIRED_FIELDS - set(header))
            if missing:
                result.fail(f"candidate gzip header is missing fields: {missing}")
            row_count = sum(1 for _ in reader)
    except (OSError, EOFError, UnicodeError, csv.Error) as exc:
        result.fail(f"could not read candidate gzip: {exc}")
        return None
    return header, row_count


def _verify_output_provenance(
    report: Mapping[str, Any], paths: Mapping[str, Path], result: GenreVerification
) -> None:
    outputs = _mapping(report, "outputs", result)
    if outputs is None:
        return
    declared_outputs = {
        "frequency_csv": paths["frequency_csv"],
        "derivational_candidates_csv_gz": paths["candidate_gzip"],
    }
    for key, actual_path in declared_outputs.items():
        declared = outputs.get(key)
        if not isinstance(declared, str) or not declared:
            result.fail(f"report.outputs.{key} must be a non-empty path string")
            continue
        actual = actual_path.resolve()
        if "\0" in declared:
            matches = False
        elif Path(declared).is_absolute():
            matches = Path(declared).resolve() == actual
        elif PureWindowsPath(declared).is_absolute():
            matches = PureWindowsPath(declared) == PureWindowsPath(str(actual))
        else:
            declared_parts = PurePosixPath(declared.replace("\\", "/")).parts
            actual_parts = PurePosixPath(str(actual).replace("\\", "/")).parts
            matches = (
                len(declared_parts) > 1
                and ".." not in declared_parts
                and actual_parts[-len(declared_parts) :] == declared_parts
            )
        if not matches:
            result.fail(f"report.outputs.{key} does not identify {actual_path.name}")


def _verify_conservation(report: Mapping[str, Any], result: GenreVerification) -> None:
    conservation = _mapping(report, "count_conservation", result)
    if conservation is None:
        return
    for flag in CONSERVATION_FLAGS:
        if conservation.get(flag) is not True:
            result.fail(f"report.count_conservation.{flag} must be true")
    numeric = ("status_total", "role_total", "token_count", "candidate_rows")
    for key in numeric:
        if not _is_nonnegative_int(conservation.get(key)):
            result.fail(f"report.count_conservation.{key} must be a non-negative integer")

    token_count = conservation.get("token_count")
    if _is_nonnegative_int(token_count) and report.get("kiwi_token_count") != token_count:
        result.fail("count_conservation.token_count does not equal kiwi_token_count")
    statuses = _nonnegative_mapping(report, "normalization_status", result)
    if statuses is not None and _is_nonnegative_int(conservation.get("status_total")):
        expected = sum(statuses.get(key, 0) for key in ("matched", "review", "unmatched"))
        if expected != conservation["status_total"]:
            result.fail("normalization_status sum does not equal status_total")
    roles = _nonnegative_mapping(report, "lexical_roles", result)
    if roles is not None and _is_nonnegative_int(conservation.get("role_total")):
        expected = sum(
            roles.get(key, 0)
            for key in (
                "direct_lexical",
                "derivational_base",
                "excluded_placeholder",
                "nonlexical",
            )
        )
        if expected != conservation["role_total"]:
            result.fail("lexical_roles sum does not equal role_total")
        if roles.get("derivational_base") != roles.get("derivational_candidate"):
            result.fail("derivational base/candidate lexical role counts differ")


def _verify_sanity(report: Mapping[str, Any], result: GenreVerification) -> None:
    units = report.get("units")
    kiwi_tokens = report.get("kiwi_token_count")
    if not _is_nonnegative_int(units):
        result.fail("report.units must be a non-negative integer")
    if not _is_nonnegative_int(kiwi_tokens):
        result.fail("report.kiwi_token_count must be a non-negative integer")
    subtype_units = _nonnegative_mapping(report, "subtype_units", result)
    if (
        subtype_units is not None
        and _is_nonnegative_int(units)
        and sum(subtype_units.values()) != units
    ):
        result.fail("subtype_units sum does not equal units")

    roles = report.get("lexical_roles")
    frequency = _nonnegative_mapping(report, "frequency", result)
    if isinstance(roles, dict) and frequency is not None:
        assignments = frequency.get("assignments")
        unique = frequency.get("unique_canonical_forms")
        direct = roles.get("direct_lexical")
        if (
            _is_nonnegative_int(assignments)
            and _is_nonnegative_int(direct)
            and assignments > direct
        ):
            result.fail("frequency assignments exceed direct lexical tokens")
        if (
            _is_nonnegative_int(unique)
            and _is_nonnegative_int(assignments)
            and unique > assignments
        ):
            result.fail("unique canonical forms exceed frequency assignments")


def _performance_summary(
    report: Mapping[str, Any], result: GenreVerification
) -> dict[str, Any] | None:
    performance = report.get("performance")
    if performance is None:
        result.warn("report.performance is absent")
        return None
    if not isinstance(performance, dict):
        result.warn("report.performance is not an object")
        return None
    summary: dict[str, Any] = {}
    for key in ("elapsed_seconds", "units_per_second", "tracemalloc_peak_mib"):
        value = performance.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            summary[key] = value
        elif value is not None:
            result.warn(f"report.performance.{key} is not a finite number")
    if "memory_note" in performance:
        summary["memory_note"] = performance["memory_note"]
    return summary


def _schema_signature(report: Mapping[str, Any]) -> dict[str, list[str]]:
    signature = {"$": sorted(report)}
    for key in FIXED_REPORT_MAPPINGS:
        value = report.get(key)
        if isinstance(value, dict):
            signature[key] = sorted(value)
    return signature


def verify_genre(output_dir: Path, genre: Genre) -> GenreVerification:
    """Verify one completed genre without writing to its output directory."""
    result = GenreVerification(genre=genre)
    paths = {
        "frequency_csv": output_dir / f"{genre}_frequency.csv",
        "candidate_gzip": output_dir / f"{genre}_derivational_candidates.csv.gz",
        "report_json": output_dir / f"{genre}_report.json",
        "checkpoint_json": output_dir / f"{genre}_checkpoint.json",
    }
    result.summary["file_sizes_bytes"] = _file_sizes(paths)
    for name in ("frequency_csv", "candidate_gzip", "report_json"):
        if not paths[name].is_file():
            result.fail(f"required output is missing: {paths[name].name}")
    if paths["checkpoint_json"].exists():
        result.fail(f"leftover checkpoint exists: {paths['checkpoint_json'].name}")
    if result.failures:
        return result

    try:
        payload = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        result.fail(f"could not read report JSON: {exc}")
        return result
    if not isinstance(payload, dict):
        result.fail("report JSON must be an object")
        return result
    report: Mapping[str, Any] = payload
    result.report_schema_signature = _schema_signature(report)
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        result.fail(
            f"report.schema_version must be {REPORT_SCHEMA_VERSION!r}, "
            f"got {report.get('schema_version')!r}"
        )
    if report.get("source") != genre:
        result.fail(f"report.source must be {genre!r}, got {report.get('source')!r}")
    if report.get("full_corpus_processed") is not True:
        result.fail("report.full_corpus_processed must be true")

    csv_stats = _verify_frequency_csv(paths["frequency_csv"], result)
    candidate_stats = _verify_candidate_gzip(paths["candidate_gzip"], result)
    frequency = report.get("frequency")
    if csv_stats is not None and isinstance(frequency, dict):
        csv_total, csv_unique = csv_stats
        if csv_total != frequency.get("assignments"):
            result.fail("frequency CSV count sum does not equal report frequency.assignments")
        if csv_unique != frequency.get("unique_canonical_forms"):
            result.fail(
                "frequency CSV row count does not equal report "
                "frequency.unique_canonical_forms"
            )
        result.summary["frequency"] = {
            "assignments": csv_total,
            "unique_canonical_forms": csv_unique,
        }
    elif not isinstance(frequency, dict):
        result.fail("report.frequency must be an object")
    if candidate_stats is not None:
        candidate_header, candidate_rows = candidate_stats
        result.summary["candidate_header_fields"] = candidate_header
        result.summary["candidate_rows"] = candidate_rows
        conservation = report.get("count_conservation")
        if (
            isinstance(conservation, dict)
            and "candidate_rows" in conservation
            and conservation["candidate_rows"] != candidate_rows
        ):
            result.fail(
                "candidate gzip row count does not equal "
                "report.count_conservation.candidate_rows"
            )

    _verify_output_provenance(report, paths, result)
    _verify_conservation(report, result)
    _verify_sanity(report, result)
    parser = report.get("parser")
    if isinstance(parser, dict):
        result.summary["schema_issues"] = parser.get("issues", {})
        if parser.get("issues"):
            result.warn("parser schema issues were reported; inspect summary.schema_issues")
    else:
        result.fail("report.parser must be an object")
    result.summary["units"] = report.get("units")
    result.summary["subtype_units"] = report.get("subtype_units")
    result.summary["lexical_roles"] = report.get("lexical_roles")
    result.summary["report_frequency"] = report.get("frequency")
    result.summary["performance"] = _performance_summary(report, result)
    return result


def verify_outputs(
    output_dir: Path, genres: Sequence[Genre] = GENRES
) -> VerificationResult:
    """Verify genres independently and compare their report schemas."""
    results = [verify_genre(output_dir, genre) for genre in genres]
    signatures = [
        result.report_schema_signature
        for result in results
        if result.report_schema_signature is not None
    ]
    schema_consistent = len(signatures) == len(results) and all(
        signature == signatures[0] for signature in signatures[1:]
    )
    failures: list[str] = []
    if len(results) > 1 and not schema_consistent:
        failures.append("genre report schemas are inconsistent")
    elif len(results) <= 1:
        schema_consistent = True
    return VerificationResult(
        genres=results,
        schema_consistent=schema_consistent,
        failures=failures,
    )


def format_human_summary(result: VerificationResult) -> str:
    """Return a concise report suitable for a terminal or CI log."""
    lines = [f"Post-run verification: {'PASS' if result.passed else 'FAIL'}"]
    for genre in result.genres:
        summary = genre.summary
        lines.append(f"\n[{genre.genre}] {'PASS' if genre.passed else 'FAIL'}")
        lines.append(f"  files: {summary.get('file_sizes_bytes', {})}")
        lines.append(
            f"  units: {summary.get('units')} subtypes: {summary.get('subtype_units')}"
        )
        lines.append(f"  frequency: {summary.get('frequency')}")
        lines.append(f"  lexical roles: {summary.get('lexical_roles')}")
        lines.append(f"  schema issues: {summary.get('schema_issues', {})}")
        lines.append(f"  performance: {summary.get('performance')}")
        lines.extend(f"  WARNING: {warning}" for warning in genre.warnings)
        lines.extend(f"  ERROR: {failure}" for failure in genre.failures)
    lines.extend(f"\nERROR: {failure}" for failure in result.failures)
    lines.append(f"\nCross-genre schema consistency: {result.schema_consistent}")
    return "\n".join(lines)
