"""Calibrate hybrid frequencies within POS/source buckets and simulate cutoffs."""

from __future__ import annotations

import contextlib
import csv
import json
import os
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from contextle_eval.lemma_frequency import normalize_word

DEFAULT_SEED = 20260823
CUTOFFS = (50, 60, 70, 80, 90)
SAMPLED_CUTOFFS = (50, 70, 90)
CONTENT_POS = frozenset({"noun", "verb", "adjective", "adverb"})
RISK_STATUSES = frozenset({"mixed_pos", "cross_pos"})
SPECIAL_REVIEW_WORDS = (
    "하다",
    "되다",
    "있다",
    "보다",
    "않다",
    "감사하다",
    "진정하다",
    "달다",
)
CALIBRATION_FIELDS = (
    "canonical_frequency_pos",
    "frequency_bucket",
    "frequency_percentile",
    "frequency_calibration_status",
    "manual_review_required",
    "high_count_risk",
)
REQUIRED_HYBRID_FIELDS = frozenset(
    {
        "word",
        "pos",
        "selected_frequency",
        "selected_frequency_source",
        "frequency_policy_source",
        "frequency_found",
        "raw_frequency",
        "lemma_frequency",
        "lemma_frequency_pos",
        "lemma_source_surface_count",
        "hybrid_risk_flags",
        "review_required",
        "review_reason",
    }
)


class FrequencyCalibrationError(RuntimeError):
    """Raised when hybrid calibration input or output is invalid."""


def canonical_pos_parts(raw_pos: str) -> set[str]:
    """Normalize candidate POS labels to mutually understandable English names."""
    mapping = {
        "명사": "noun",
        "고유 명사": "noun",
        "고유명사": "noun",
        "의존 명사": "noun",
        "의존명사": "noun",
        "대명사": "pronoun",
        "수사": "numeral",
        "부사": "adverb",
        "관형사": "determiner",
        "감탄사": "interjection",
        "동사": "verb",
        "자동사": "verb",
        "타동사": "verb",
        "조동사": "verb",
        "보조 동사": "verb",
        "보조동사": "verb",
        "형용사": "adjective",
        "보조 형용사": "adjective",
        "보조형용사": "adjective",
    }
    return {mapping[pos] for pos in raw_pos.split("|") if pos in mapping}


def _lemma_pos_parts(raw_pos: str) -> set[str]:
    mapping = {
        "명사": "noun",
        "고유명사": "noun",
        "의존명사": "noun",
        "대명사": "pronoun",
        "수사": "numeral",
        "부사": "adverb",
        "관형사": "determiner",
        "감탄사": "interjection",
        "동사": "verb",
        "형용사": "adjective",
    }
    return {mapping[pos] for pos in raw_pos.split("|") if pos in mapping}


def _bucket_for_row(row: dict[str, str]) -> tuple[str, str, bool, bool]:
    """Return canonical POS, bucket, mixed flag, and cross-POS flag."""
    candidate_parts = canonical_pos_parts(row.get("pos", ""))
    policy_source = row.get("frequency_policy_source", "")
    hybrid_flags = set(row.get("hybrid_risk_flags", "").split("|")) - {""}
    cross = "cross_pos_lemma_buckets" in hybrid_flags
    mixed = "mixed_candidate_pos_predicate_precedence" in hybrid_flags

    if policy_source == "lemma":
        matched_parts = _lemma_pos_parts(row.get("lemma_frequency_pos", ""))
        predicate_parts = candidate_parts & {"verb", "adjective"}
        selected_parts = matched_parts or predicate_parts
        if len(matched_parts) > 1:
            cross = True
        if len(predicate_parts) > 1:
            cross = True
        if candidate_parts - {"verb", "adjective"}:
            mixed = True
        canonical = "+".join(sorted(selected_parts)) or "predicate_other"
        prefix = "cross" if cross else canonical
        return canonical, f"lemma:{prefix}", mixed, cross

    non_predicate = candidate_parts - {"verb", "adjective"}
    canonical = "+".join(sorted(non_predicate)) or "other"
    if len(non_predicate) > 1:
        mixed = True
        return canonical, f"raw:mixed:{canonical}", mixed, cross
    return canonical, f"raw:{canonical}", mixed, cross


def percentile_by_frequency(frequencies: Sequence[int]) -> dict[int, float]:
    """Return empirical-CDF percentiles: 100 * count(value <= x) / N."""
    if not frequencies:
        return {}
    counts = Counter(frequencies)
    cumulative = 0
    total = len(frequencies)
    result: dict[int, float] = {}
    for frequency in sorted(counts):
        cumulative += counts[frequency]
        result[frequency] = 100 * cumulative / total
    return result


def _parse_frequency(raw: str, *, row_number: int) -> int | None:
    if not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise FrequencyCalibrationError(
            f"Hybrid row {row_number} has invalid selected_frequency {raw!r}."
        ) from exc
    if value < 0:
        raise FrequencyCalibrationError(
            f"Hybrid row {row_number} has negative selected_frequency {value}."
        )
    return value


def load_hybrid_candidates(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        handle = path.open(encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise FrequencyCalibrationError(
            f"Could not read hybrid CSV {path}: {exc}"
        ) from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_HYBRID_FIELDS <= set(
            reader.fieldnames
        ):
            raise FrequencyCalibrationError(
                f"Hybrid CSV requires fields {sorted(REQUIRED_HYBRID_FIELDS)}."
            )
        collisions = set(reader.fieldnames) & set(CALIBRATION_FIELDS)
        if collisions:
            raise FrequencyCalibrationError(
                f"Hybrid CSV already has calibration fields: {sorted(collisions)}."
            )
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames)
    if not rows:
        raise FrequencyCalibrationError("Hybrid CSV contains no candidate rows.")
    return fieldnames, rows


def calibrate_candidates(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """Assign a bucket and within-bucket percentile without changing frequencies."""
    calibrated: list[dict[str, str]] = []
    by_bucket: defaultdict[str, list[int]] = defaultdict(list)
    parsed_values: list[int | None] = []
    bucket_meta: list[tuple[str, str, bool, bool]] = []

    for row_number, source in enumerate(rows, start=2):
        row = dict(source)
        found = row.get("frequency_found") == "true"
        frequency = _parse_frequency(
            row.get("selected_frequency", ""), row_number=row_number
        )
        if found != (frequency is not None):
            raise FrequencyCalibrationError(
                f"Hybrid row {row_number} has inconsistent frequency_found/selected_frequency."
            )
        canonical, bucket, mixed, cross = _bucket_for_row(row)
        calibrated.append(row)
        parsed_values.append(frequency)
        bucket_meta.append((canonical, bucket, mixed, cross))
        if frequency is not None:
            by_bucket[bucket].append(frequency)

    percentile_maps = {
        bucket: percentile_by_frequency(frequencies)
        for bucket, frequencies in by_bucket.items()
    }
    for row, frequency, (canonical, bucket, mixed, cross) in zip(
        calibrated, parsed_values, bucket_meta, strict=True
    ):
        percentile = (
            percentile_maps[bucket][frequency] if frequency is not None else None
        )
        source_surface_count = int(row.get("lemma_source_surface_count") or 0)
        high_count = bool(
            frequency is not None
            and row.get("selected_frequency_source") == "lemma"
            and percentile is not None
            and percentile >= 99
            and source_surface_count >= 25
        )
        special = normalize_word(row.get("word", "")) in SPECIAL_REVIEW_WORDS
        existing_risk = bool(row.get("hybrid_risk_flags", ""))
        if frequency is None:
            status = "unmatched"
        elif cross:
            status = "cross_pos"
        elif mixed:
            status = "mixed_pos"
        elif high_count:
            status = "high_count_risk"
        else:
            status = "normal"
        manual = (
            special
            or existing_risk
            or status
            in {
                "mixed_pos",
                "cross_pos",
                "high_count_risk",
            }
        )
        row.update(
            {
                "canonical_frequency_pos": canonical,
                "frequency_bucket": bucket,
                "frequency_percentile": (
                    f"{percentile:.6f}" if percentile is not None else ""
                ),
                "frequency_calibration_status": status,
                "manual_review_required": "true" if manual else "false",
                "high_count_risk": "true" if high_count else "false",
            }
        )
    return calibrated


def _atomic_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise FrequencyCalibrationError(
            f"Could not write calibrated CSV {path}: {exc}"
        ) from exc


def write_calibrated_csv(
    path: Path, input_fields: Sequence[str], rows: Sequence[dict[str, str]]
) -> None:
    _atomic_csv(path, (*input_fields, *CALIBRATION_FIELDS), rows)


def _nearest_percentile(values: Sequence[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, (len(ordered) * percentile + 99) // 100 - 1))
    return ordered[index]


def bucket_distributions(rows: Sequence[dict[str, str]]) -> dict[str, dict[str, Any]]:
    buckets: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[row["frequency_bucket"]].append(row)
    report: dict[str, dict[str, Any]] = {}
    for bucket, members in sorted(buckets.items()):
        matched = [row for row in members if row["frequency_found"] == "true"]
        frequencies = [int(row["selected_frequency"]) for row in matched]
        percentiles = [float(row["frequency_percentile"]) for row in matched]
        bands = Counter()
        for value in percentiles:
            if value < 10:
                bands["0_to_lt_10"] += 1
            elif value < 25:
                bands["10_to_lt_25"] += 1
            elif value < 50:
                bands["25_to_lt_50"] += 1
            elif value < 70:
                bands["50_to_lt_70"] += 1
            elif value < 90:
                bands["70_to_lt_90"] += 1
            else:
                bands["90_to_100"] += 1
        report[bucket] = {
            "candidate_count": len(members),
            "matched_count": len(matched),
            "unmatched_count": len(members) - len(matched),
            "frequency_min": min(frequencies) if frequencies else None,
            "frequency_max": max(frequencies) if frequencies else None,
            "frequency_median": statistics.median(frequencies) if frequencies else None,
            "percentile_min": min(percentiles) if percentiles else None,
            "percentile_p25": _nearest_percentile(percentiles, 25),
            "percentile_median": _nearest_percentile(percentiles, 50),
            "percentile_p75": _nearest_percentile(percentiles, 75),
            "percentile_max": max(percentiles) if percentiles else None,
            "percentile_bands": dict(sorted(bands.items())),
        }
    return report


def _ratio(count: int, total: int) -> float:
    return count / total if total else 0.0


def _simulation_stats(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    total = len(rows)
    canonical = Counter(row["canonical_frequency_pos"] for row in rows)
    sources = Counter(row["selected_frequency_source"] for row in rows)
    reviewed = sum(row.get("review_required", "").lower() == "true" for row in rows)
    manual = sum(row["manual_review_required"] == "true" for row in rows)
    high_count = sum(row["high_count_risk"] == "true" for row in rows)
    reason_stats = {}
    for reason in (
        "technical_term",
        "historical_term",
        "archaic",
        "long_word",
        "explicit_rare_label",
    ):
        count = sum(reason in row.get("review_reason", "").split("|") for row in rows)
        reason_stats[reason] = {"count": count, "ratio": _ratio(count, total)}
    return {
        "retained_matched_count": total,
        "canonical_pos_counts": dict(sorted(canonical.items())),
        "canonical_pos_ratios": {
            key: _ratio(value, total) for key, value in sorted(canonical.items())
        },
        "source_counts": dict(sorted(sources.items())),
        "source_ratios": {
            key: _ratio(value, total) for key, value in sorted(sources.items())
        },
        "review_required_count": reviewed,
        "review_required_ratio": _ratio(reviewed, total),
        "not_review_required_count": total - reviewed,
        "not_review_required_ratio": _ratio(total - reviewed, total),
        "review_reason": reason_stats,
        "manual_review_required_count": manual,
        "manual_review_required_ratio": _ratio(manual, total),
        "high_count_risk_count": high_count,
        "high_count_risk_ratio": _ratio(high_count, total),
    }


def _scenario_rows(
    rows: Sequence[dict[str, str]], *, content_only: bool, exclude_risk: bool
) -> list[dict[str, str]]:
    selected = [row for row in rows if row["frequency_found"] == "true"]
    if content_only:
        selected = [
            row
            for row in selected
            if (parts := canonical_pos_parts(row.get("pos", "")))
            and parts <= CONTENT_POS
        ]
    if exclude_risk:
        selected = [
            row
            for row in selected
            if row["frequency_calibration_status"] not in RISK_STATUSES
        ]
    return selected


def simulate_cutoffs(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    scenarios = {
        "all_pos_include_risk": (False, False),
        "all_pos_exclude_mixed_cross": (False, True),
        "content_words_include_risk": (True, False),
        "content_words_exclude_mixed_cross": (True, True),
    }
    result: dict[str, Any] = {}
    for name, (content_only, exclude_risk) in scenarios.items():
        eligible = _scenario_rows(
            rows, content_only=content_only, exclude_risk=exclude_risk
        )
        result[name] = {
            f"P{cutoff}": _simulation_stats(
                [
                    row
                    for row in eligible
                    if float(row["frequency_percentile"]) >= cutoff
                ]
            )
            for cutoff in CUTOFFS
        }
    return result


def _sample_view(row: dict[str, str]) -> dict[str, str]:
    return {
        key: row.get(key, "")
        for key in (
            "word",
            "pos",
            "selected_frequency",
            "selected_frequency_source",
            "canonical_frequency_pos",
            "frequency_bucket",
            "frequency_percentile",
            "frequency_calibration_status",
            "review_required",
            "review_reason",
            "manual_review_required",
            "hybrid_risk_flags",
        )
    }


def _random_sample(
    rows: Sequence[dict[str, str]], *, seed: int, size: int
) -> list[dict[str, str]]:
    if not rows:
        return []
    chosen = random.Random(seed).sample(list(rows), min(size, len(rows)))
    return [_sample_view(row) for row in chosen]


def build_quality_samples(
    rows: Sequence[dict[str, str]], *, seed: int
) -> dict[str, Any]:
    scenarios = {
        "all_pos_include_risk": (False, False),
        "all_pos_exclude_mixed_cross": (False, True),
        "content_words_include_risk": (True, False),
        "content_words_exclude_mixed_cross": (True, True),
    }
    report: dict[str, Any] = {}
    seed_offset = 0
    for scenario, (content_only, exclude_risk) in scenarios.items():
        eligible = _scenario_rows(
            rows, content_only=content_only, exclude_risk=exclude_risk
        )
        report[scenario] = {}
        for cutoff in SAMPLED_CUTOFFS:
            retained = [
                row for row in eligible if float(row["frequency_percentile"]) >= cutoff
            ]
            above = sorted(
                retained,
                key=lambda row: (float(row["frequency_percentile"]), row["word"]),
            )[:50]
            below = sorted(
                (
                    row
                    for row in eligible
                    if float(row["frequency_percentile"]) < cutoff
                ),
                key=lambda row: (-float(row["frequency_percentile"]), row["word"]),
            )[:50]
            samples: dict[str, Any] = {
                "retained_random_100": _random_sample(
                    retained, seed=seed + seed_offset, size=100
                ),
                "just_above_50": [_sample_view(row) for row in above],
                "just_below_50": [_sample_view(row) for row in below],
            }
            seed_offset += 1
            for pos in ("noun", "verb", "adjective", "adverb"):
                pool = [
                    row
                    for row in retained
                    if pos in canonical_pos_parts(row.get("pos", ""))
                ]
                samples[f"{pos}_50"] = _random_sample(
                    pool, seed=seed + seed_offset, size=50
                )
                seed_offset += 1
            reviewed = [
                row
                for row in retained
                if row.get("review_required", "").lower() == "true"
            ]
            samples["review_required_50"] = _random_sample(
                reviewed, seed=seed + seed_offset, size=50
            )
            seed_offset += 1
            samples["manual_review_all"] = [
                _sample_view(row)
                for row in retained
                if row["manual_review_required"] == "true"
            ]
            report[scenario][f"P{cutoff}"] = samples
    return report


def percentile_extremes(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    matched = [row for row in rows if row["frequency_found"] == "true"]
    for pos in sorted(
        CONTENT_POS | {"determiner", "interjection", "pronoun", "numeral"}
    ):
        pool = [
            row for row in matched if pos in canonical_pos_parts(row.get("pos", ""))
        ]
        high = sorted(
            pool,
            key=lambda row: (-float(row["frequency_percentile"]), row["word"]),
        )[:30]
        low = sorted(
            pool,
            key=lambda row: (float(row["frequency_percentile"]), row["word"]),
        )[:30]
        report[pos] = {
            "highest_30": [_sample_view(row) for row in high],
            "lowest_30": [_sample_view(row) for row in low],
        }
    return report


def special_word_report(
    rows: Sequence[dict[str, str]], simulations: dict[str, Any]
) -> dict[str, Any]:
    by_word = {row["word"]: row for row in rows}
    report: dict[str, Any] = {}
    for word in SPECIAL_REVIEW_WORDS:
        row = by_word.get(word)
        if row is None:
            report[word] = {"candidate_present": False}
            continue
        outcomes: dict[str, dict[str, str]] = {}
        for scenario_name in simulations:
            content_only = scenario_name.startswith("content_words")
            exclude_risk = scenario_name.endswith("exclude_mixed_cross")
            content_allowed = (
                bool(parts := canonical_pos_parts(row.get("pos", "")))
                and parts <= CONTENT_POS
            )
            outcomes[scenario_name] = {}
            for cutoff in CUTOFFS:
                if row["frequency_found"] != "true":
                    outcome = "unmatched"
                elif content_only and not content_allowed:
                    outcome = "not_in_pos_scenario"
                elif (
                    exclude_risk
                    and row["frequency_calibration_status"] in RISK_STATUSES
                ):
                    outcome = "excluded_risk"
                elif float(row["frequency_percentile"]) >= cutoff:
                    outcome = "meets_threshold_manual_review"
                else:
                    outcome = "below_threshold_manual_review"
                outcomes[scenario_name][f"P{cutoff}"] = outcome
        report[word] = {
            "candidate_present": True,
            "selected_frequency": row["selected_frequency"],
            "selected_frequency_source": row["selected_frequency_source"],
            "pos": row["pos"],
            "frequency_bucket": row["frequency_bucket"],
            "frequency_percentile": row["frequency_percentile"],
            "source_surface_count": row["lemma_source_surface_count"],
            "risk_flags": row["hybrid_risk_flags"],
            "calibration_status": row["frequency_calibration_status"],
            "manual_review_required": row["manual_review_required"],
            "scenario_outcomes": outcomes,
        }
    return report


def build_calibration_report(
    rows: Sequence[dict[str, str]], *, seed: int
) -> dict[str, Any]:
    simulations = simulate_cutoffs(rows)
    status_counts = Counter(row["frequency_calibration_status"] for row in rows)
    matched = sum(row["frequency_found"] == "true" for row in rows)
    mixed = [
        row
        for row in rows
        if "mixed_candidate_pos_predicate_precedence"
        in row.get("hybrid_risk_flags", "").split("|")
    ]
    cross = [
        row
        for row in rows
        if row["frequency_calibration_status"] == "cross_pos"
        or "cross_pos_lemma_buckets" in row.get("hybrid_risk_flags", "").split("|")
    ]
    multiple_lemma_pos = [
        row
        for row in rows
        if "cross_pos_lemma_buckets" in row.get("hybrid_risk_flags", "").split("|")
    ]
    high_count = [row for row in rows if row["high_count_risk"] == "true"]
    return {
        "formula": (
            "frequency_percentile = 100 * count(bucket matched candidates with "
            "selected_frequency <= current selected_frequency) / bucket matched count"
        ),
        "tie_policy": "equal selected_frequency in one bucket receives equal percentile",
        "percentile_direction": "higher frequency produces an equal or higher percentile",
        "candidate_count": len(rows),
        "matched_count": matched,
        "unmatched_count": len(rows) - matched,
        "calibration_status_counts": dict(sorted(status_counts.items())),
        "bucket_distributions": bucket_distributions(rows),
        "cutoff_simulations": simulations,
        "quality_samples": build_quality_samples(rows, seed=seed),
        "pos_percentile_extremes": percentile_extremes(rows),
        "special_word_review": special_word_report(rows, simulations),
        "mixed_pos_count": len(mixed),
        "mixed_pos_all": [_sample_view(row) for row in mixed],
        "cross_pos_count": len(cross),
        "cross_pos_all": [_sample_view(row) for row in cross],
        "multiple_lemma_pos_bucket_count": len(multiple_lemma_pos),
        "multiple_lemma_pos_bucket_all": [
            _sample_view(row) for row in multiple_lemma_pos
        ],
        "high_count_risk_count": len(high_count),
        "high_count_risk_all": [_sample_view(row) for row in high_count],
        "unmatched_policy": {
            "frequency_percentile": None,
            "frequency_calibration_status": "unmatched",
            "treated_as_zero": False,
            "automatically_removed": False,
            "included_in_cutoff_simulation": False,
        },
        "sample_seed": seed,
    }


def format_calibration_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2)
