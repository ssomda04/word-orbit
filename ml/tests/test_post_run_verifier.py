"""Tests for read-only raw-frequency post-run verification."""

from __future__ import annotations

import csv
import gzip
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from contextle_eval.post_run_verifier import GENRES, verify_genre, verify_outputs

CANDIDATE_FIELDS = (
    "source",
    "source_text_id",
    "canonical_form",
    "frequency_assignment",
)


def _load_cli() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts/verify_modu_raw_frequency.py"
    spec = importlib.util.spec_from_file_location("verify_modu_raw_frequency", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLI = _load_cli()


def _report(genre: str) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "source": genre,
        "kiwipiepy_version": "test",
        "full_corpus_processed": True,
        "input_archives": [f"{genre}.zip"],
        "units": 2,
        "subtype_units": {genre: 2},
        "kiwi_token_count": 5,
        "normalization_status": {"matched": 3, "review": 1, "unmatched": 1},
        "lexical_roles": {
            "direct_lexical": 3,
            "derivational_base": 1,
            "derivational_candidate": 1,
            "excluded_placeholder": 0,
            "nonlexical": 1,
        },
        "frequency": {
            "assignments": 3,
            "unique_canonical_forms": 2,
            "derivational_base_assignments": 0,
            "derivational_candidate_assignments": 0,
        },
        "major_pos_and_suffix_distribution": {"NNG": 3},
        "placeholder_audit": {
            "occurrences": 0,
            "units": 0,
            "tokens_excluded": 0,
            "candidates_excluded": 0,
        },
        "markup_audit": {
            "units": 0,
            "removed_tag_count": 0,
            "units_without_hangul": 0,
        },
        "parser": {"schema_paths": {"fixture": 2}, "issues": {}},
        "count_conservation": {
            "status_total": 5,
            "role_total": 5,
            "token_count": 5,
            "status_conserved": True,
            "roles_conserved": True,
            "base_equals_candidate_count": True,
            "candidate_rows": 1,
        },
        "performance": {
            "elapsed_seconds": 1.5,
            "units_per_second": 1.333,
            "tracemalloc_peak_mib": 2.0,
            "memory_note": "fixture",
        },
        "outputs": {
            "frequency_csv": f"{genre}_frequency.csv",
            "derivational_candidates_csv_gz": (
                f"{genre}_derivational_candidates.csv.gz"
            ),
        },
    }


def _write_complete(root: Path, genre: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    frequency = root / f"{genre}_frequency.csv"
    frequency.write_text("canonical_form,count\n가,1\n나,2\n", encoding="utf-8")
    candidate = root / f"{genre}_derivational_candidates.csv.gz"
    with gzip.open(candidate, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "source": genre,
                "source_text_id": "1",
                "canonical_form": "하다",
                "frequency_assignment": "0",
            }
        )
    report = _report(genre)
    report["outputs"] = {
        "frequency_csv": f"{root.name}/{genre}_frequency.csv",
        "derivational_candidates_csv_gz": (
            f"{root.name}/{genre}_derivational_candidates.csv.gz"
        ),
    }
    (root / f"{genre}_report.json").write_text(
        json.dumps(report, ensure_ascii=False), encoding="utf-8"
    )


def _mutate_report(root: Path, genre: str, mutation: object) -> None:
    path = root / f"{genre}_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert callable(mutation)
    mutation(report)
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")


def test_synthetic_complete_results_pass_and_summarize(tmp_path: Path) -> None:
    for genre in GENRES:
        _write_complete(tmp_path, genre)
    _mutate_report(
        tmp_path,
        "dialogue",
        lambda report: report["parser"]["issues"].update(blank_form=1),
    )

    result = verify_outputs(tmp_path)

    assert result.passed is True
    assert result.schema_consistent is True
    assert result.genres[0].summary["frequency"] == {
        "assignments": 3,
        "unique_canonical_forms": 2,
    }
    assert result.genres[1].warnings


def test_missing_file_and_leftover_checkpoint_are_blocking(tmp_path: Path) -> None:
    _write_complete(tmp_path, "newspaper")
    (tmp_path / "newspaper_frequency.csv").unlink()
    (tmp_path / "newspaper_checkpoint.json").write_text("{}", encoding="utf-8")

    result = verify_genre(tmp_path, "newspaper")

    assert result.passed is False
    assert any("required output is missing" in failure for failure in result.failures)
    assert any("leftover checkpoint" in failure for failure in result.failures)


@pytest.mark.parametrize(
    ("frequency_text", "expected"),
    [
        ("canonical_form,count\n가,1\n나,1\n", "count sum"),
        ("canonical_form,count\n가,1\n가,2\n", "duplicate canonical form"),
    ],
)
def test_frequency_mismatch_and_duplicate_are_blocking(
    tmp_path: Path, frequency_text: str, expected: str
) -> None:
    _write_complete(tmp_path, "dialogue")
    (tmp_path / "dialogue_frequency.csv").write_text(
        frequency_text, encoding="utf-8"
    )

    result = verify_genre(tmp_path, "dialogue")

    assert result.passed is False
    assert any(expected in failure for failure in result.failures)


@pytest.mark.parametrize(
    ("frequency_text", "expected"),
    [
        ("canonical_form,count\n가\n", "missing or blank count"),
        ('canonical_form,count\n"가,1\n', "could not read frequency CSV"),
        ("canonical_form,count\n가,1,extra\n", "extra columns"),
        ("canonical_form,count\n가,not-an-integer\n", "invalid count"),
    ],
)
def test_malformed_frequency_rows_are_blocking_without_exception_leaks(
    tmp_path: Path, frequency_text: str, expected: str
) -> None:
    _write_complete(tmp_path, "dialogue")
    (tmp_path / "dialogue_frequency.csv").write_text(
        frequency_text, encoding="utf-8"
    )

    result = verify_genre(tmp_path, "dialogue")

    assert result.passed is False
    assert any(expected in failure for failure in result.failures)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda report: report.update(source="dialogue"), "report.source"),
        (lambda report: report.update(schema_version="2.0"), "schema_version"),
        (
            lambda report: report.update(full_corpus_processed=False),
            "full_corpus_processed",
        ),
    ],
)
def test_wrong_source_schema_and_incomplete_are_blocking(
    tmp_path: Path, mutation: object, expected: str
) -> None:
    _write_complete(tmp_path, "newspaper")
    _mutate_report(tmp_path, "newspaper", mutation)

    result = verify_genre(tmp_path, "newspaper")

    assert result.passed is False
    assert any(expected in failure for failure in result.failures)


def test_cross_genre_schema_mismatch_is_blocking(tmp_path: Path) -> None:
    for genre in GENRES:
        _write_complete(tmp_path, genre)
    _mutate_report(
        tmp_path,
        "online",
        lambda report: report.update(unexpected_future_field=True),
    )

    result = verify_outputs(tmp_path)

    assert all(genre.passed for genre in result.genres)
    assert result.schema_consistent is False
    assert result.passed is False
    assert result.failures == ["genre report schemas are inconsistent"]


def test_false_count_conservation_flag_is_blocking(tmp_path: Path) -> None:
    _write_complete(tmp_path, "online")
    _mutate_report(
        tmp_path,
        "online",
        lambda report: report["count_conservation"].update(roles_conserved=False),
    )

    result = verify_genre(tmp_path, "online")

    assert result.passed is False
    assert any("roles_conserved" in failure for failure in result.failures)


def test_candidate_header_and_expected_row_pass(tmp_path: Path) -> None:
    _write_complete(tmp_path, "dialogue")

    result = verify_genre(tmp_path, "dialogue")

    assert result.passed is True
    assert result.summary["candidate_rows"] == 1


def test_candidate_header_only_with_expected_row_is_blocking(tmp_path: Path) -> None:
    _write_complete(tmp_path, "dialogue")
    candidate = tmp_path / "dialogue_derivational_candidates.csv.gz"
    with gzip.open(candidate, "wt", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS).writeheader()

    result = verify_genre(tmp_path, "dialogue")

    assert result.passed is False
    assert any("candidate gzip row count" in failure for failure in result.failures)


def test_candidate_row_count_mismatch_is_blocking(tmp_path: Path) -> None:
    _write_complete(tmp_path, "dialogue")
    _mutate_report(
        tmp_path,
        "dialogue",
        lambda report: report["count_conservation"].update(candidate_rows=2),
    )

    result = verify_genre(tmp_path, "dialogue")

    assert result.passed is False
    assert any("candidate gzip row count" in failure for failure in result.failures)


def test_truncated_candidate_gzip_is_blocking(tmp_path: Path) -> None:
    _write_complete(tmp_path, "dialogue")
    candidate = tmp_path / "dialogue_derivational_candidates.csv.gz"
    candidate.write_bytes(candidate.read_bytes()[:-4])

    result = verify_genre(tmp_path, "dialogue")

    assert result.passed is False
    assert any("could not read candidate gzip" in failure for failure in result.failures)


@pytest.mark.parametrize(
    ("key", "wrong_path"),
    [
        ("frequency_csv", "wrong_frequency.csv"),
        ("derivational_candidates_csv_gz", "wrong_candidates.csv.gz"),
    ],
)
def test_wrong_output_provenance_is_blocking(
    tmp_path: Path, key: str, wrong_path: str
) -> None:
    _write_complete(tmp_path, "dialogue")
    _mutate_report(
        tmp_path,
        "dialogue",
        lambda report: report["outputs"].update({key: wrong_path}),
    )

    result = verify_genre(tmp_path, "dialogue")

    assert result.passed is False
    assert any(f"report.outputs.{key}" in failure for failure in result.failures)


@pytest.mark.parametrize("separator", ["\\", "/"])
def test_production_relative_output_provenance_passes(
    tmp_path: Path, separator: str
) -> None:
    output_dir = tmp_path / "ml" / "data" / "modu_raw_frequency"
    _write_complete(output_dir, "dialogue")
    prefix = separator.join(("ml", "data", "modu_raw_frequency"))
    _mutate_report(
        output_dir,
        "dialogue",
        lambda report: report["outputs"].update(
            frequency_csv=f"{prefix}{separator}dialogue_frequency.csv",
            derivational_candidates_csv_gz=(
                f"{prefix}{separator}dialogue_derivational_candidates.csv.gz"
            ),
        ),
    )

    assert verify_genre(output_dir, "dialogue").passed is True


@pytest.mark.parametrize(
    "declared",
    [
        "ml/data/wrong/dialogue_frequency.csv",
        "wrong/dialogue_frequency.csv",
        "ml/data/modu_raw_frequency/wrong.csv",
        "dialogue_frequency.csv",
        "../modu_raw_frequency/dialogue_frequency.csv",
    ],
)
def test_invalid_relative_frequency_provenance_is_blocking(
    tmp_path: Path, declared: str
) -> None:
    output_dir = tmp_path / "ml" / "data" / "modu_raw_frequency"
    _write_complete(output_dir, "dialogue")
    _mutate_report(
        output_dir,
        "dialogue",
        lambda report: report["outputs"].update(frequency_csv=declared),
    )

    result = verify_genre(output_dir, "dialogue")

    assert result.passed is False
    assert any("report.outputs.frequency_csv" in item for item in result.failures)


def test_absolute_output_provenance_passes(tmp_path: Path) -> None:
    _write_complete(tmp_path, "dialogue")
    _mutate_report(
        tmp_path,
        "dialogue",
        lambda report: report["outputs"].update(
            frequency_csv=str((tmp_path / "dialogue_frequency.csv").resolve()),
            derivational_candidates_csv_gz=str(
                (tmp_path / "dialogue_derivational_candidates.csv.gz").resolve()
            ),
        ),
    )

    assert verify_genre(tmp_path, "dialogue").passed is True


def test_incorrect_absolute_output_provenance_is_blocking(tmp_path: Path) -> None:
    _write_complete(tmp_path, "dialogue")
    _mutate_report(
        tmp_path,
        "dialogue",
        lambda report: report["outputs"].update(
            frequency_csv=str((tmp_path / "wrong_frequency.csv").resolve())
        ),
    )

    result = verify_genre(tmp_path, "dialogue")

    assert result.passed is False
    assert any("report.outputs.frequency_csv" in item for item in result.failures)


def test_json_output_cannot_overwrite_production_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_complete(tmp_path, "dialogue")
    report = tmp_path / "dialogue_report.json"
    original = report.read_bytes()

    exit_code = CLI.main(
        [
            "--output-dir",
            str(tmp_path),
            "--genre",
            "dialogue",
            "--json-output",
            str(report),
        ]
    )

    assert exit_code == 2
    assert report.read_bytes() == original
    assert "strictly read-only" in capsys.readouterr().err


def test_json_output_inside_output_dir_is_rejected(tmp_path: Path) -> None:
    output_dir = tmp_path / "production"
    _write_complete(output_dir, "dialogue")
    json_output = output_dir / "new-summary.json"

    exit_code = CLI.main(
        [
            "--output-dir",
            str(output_dir),
            "--genre",
            "dialogue",
            "--json-output",
            str(json_output),
        ]
    )

    assert exit_code == 2
    assert not json_output.exists()


def test_relative_json_output_collision_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "production"
    _write_complete(output_dir, "dialogue")
    monkeypatch.chdir(tmp_path)

    exit_code = CLI.main(
        [
            "--output-dir",
            "production",
            "--genre",
            "dialogue",
            "--json-output",
            "production/relative-summary.json",
        ]
    )

    assert exit_code == 2
    assert not (output_dir / "relative-summary.json").exists()


def test_cli_exit_codes_human_summary_and_optional_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for genre in GENRES:
        _write_complete(tmp_path, genre)
    json_output = tmp_path.parent / "verification-summary.json"

    exit_code = CLI.main(
        ["--output-dir", str(tmp_path), "--json-output", str(json_output)]
    )

    assert exit_code == 0
    assert "Post-run verification: PASS" in capsys.readouterr().out
    assert json.loads(json_output.read_text(encoding="utf-8"))["passed"] is True

    (tmp_path / "online_checkpoint.json").write_text("{}", encoding="utf-8")
    assert CLI.main(["--output-dir", str(tmp_path)]) == 1
