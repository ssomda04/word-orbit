#!/usr/bin/env python3
"""Run bounded Kiwi normalization pilots over three raw NIKL corpora."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import tracemalloc
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextle_eval.modu_baseform_analysis import (
    load_answer_candidates,
    load_vocabulary,
)
from contextle_eval.modu_kiwi_adapter import (
    MAJOR_CONTENT_POS,
    KiwiAdapterError,
    adapt_kiwi_tokens,
    extract_derivational_candidates,
)
from contextle_eval.modu_raw_corpus import RawSource, iter_raw_text_units

DEFAULT_ROOT = Path(r"C:\data\modu_corpus")
PLACEHOLDER_PATTERN = re.compile(
    r"&(?:company|brand|name)&|(?<![A-Za-z])(?:company|brand|name)(?![A-Za-z])",
    re.IGNORECASE,
)
PLACEHOLDER_FORMS = frozenset({"company", "brand", "name"})
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
HANGUL_PATTERN = re.compile(r"[가-힣]")
EXAMPLE_LIMIT = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--newspaper-limit", type=int, default=500)
    parser.add_argument("--dialogue-limit", type=int, default=500)
    parser.add_argument("--online-limit-per-subtype", type=int, default=250)
    parser.add_argument("--vocabulary", type=Path, default=ML_ROOT / "data/game_words.txt")
    parser.add_argument(
        "--answer-candidates",
        type=Path,
        default=ML_ROOT / "data/answer_candidates.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ML_ROOT / "data/modu_raw_kiwi_pilot.json",
    )
    return parser.parse_args()


def _append_example(target: list[dict[str, Any]], value: dict[str, Any]) -> None:
    if len(target) < EXAMPLE_LIMIT:
        target.append(value)


def run_source(
    *,
    zip_path: Path,
    source: RawSource,
    limit: int,
    kiwi: Any,
    game_words: frozenset[str],
    answer_words: frozenset[str],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    pos_counts: Counter[str] = Counter()
    schema_paths: Counter[str] = Counter()
    normalization_examples: list[dict[str, Any]] = []
    candidate_examples: list[dict[str, Any]] = []
    placeholder_examples: list[dict[str, Any]] = []
    started = time.perf_counter()
    tracemalloc.start()
    for unit in iter_raw_text_units(zip_path, source=source, limit=limit):
        counts["text_units"] += 1
        schema_paths[unit.schema_path] += 1
        placeholder_matches = list(PLACEHOLDER_PATTERN.finditer(unit.form))
        html_tags = HTML_TAG_PATTERN.findall(unit.form)
        if html_tags:
            counts["html_tag_occurrences"] += len(html_tags)
            counts["text_units_with_html_tags"] += 1
        if not HANGUL_PATTERN.search(unit.form):
            counts["text_units_without_hangul"] += 1
        if placeholder_matches:
            counts["placeholder_occurrences"] += len(placeholder_matches)
            counts["text_units_with_placeholders"] += 1
            _append_example(
                placeholder_examples,
                {
                    "source_text_id": unit.text_id,
                    "markers": [match.group() for match in placeholder_matches],
                },
            )
        raw_tokens = list(kiwi.tokenize(unit.form))
        try:
            records = adapt_kiwi_tokens(
                unit.text_id,
                unit.form,
                raw_tokens,
                game_words,
                answer_words,
            )
        except KiwiAdapterError as exc:
            invalid = [
                {
                    "form": token.form,
                    "tag": token.tag,
                    "start": token.start,
                    "len": token.len,
                    "word_position": token.word_position,
                    "sent_position": token.sent_position,
                }
                for token in raw_tokens
                if token.start < 0
                or token.len < 0
                or token.word_position < 0
                or token.sent_position < 0
            ]
            raise KiwiAdapterError(
                f"{source} {unit.text_id}: {exc}; invalid={invalid[:3]}"
            ) from exc
        candidates = extract_derivational_candidates(records, game_words, answer_words)
        counts["kiwi_tokens"] += len(records)
        counts["derivational_candidates"] += len(candidates)
        for record in records:
            is_placeholder = record.source_morpheme.casefold() in PLACEHOLDER_FORMS
            if is_placeholder:
                counts["placeholder_tokens_excluded"] += 1
                status_counts["unmatched"] += 1
            else:
                status_counts[record.status] += 1
            if record.source_pos in MAJOR_CONTENT_POS:
                pos_counts[record.source_pos] += 1
            if record.source_pos in {"XSV", "XSA", "VX"}:
                pos_counts[record.source_pos] += 1
            if record.source_pos in MAJOR_CONTENT_POS:
                _append_example(
                    normalization_examples,
                    {
                        "source_text_id": unit.text_id,
                        "eojeol_surface": record.eojeol_surface,
                        "morpheme": record.source_morpheme,
                        "pos": record.source_pos,
                        "canonical_form": record.canonical_form,
                        "status": "unmatched" if is_placeholder else record.status,
                    },
                )
        for candidate in candidates:
            if candidate.base_form.casefold() in PLACEHOLDER_FORMS:
                counts["placeholder_candidates_excluded"] += 1
                continue
            _append_example(candidate_examples, asdict(candidate))
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = time.perf_counter() - started
    return {
        "limit": limit,
        "analyzed_text_units": counts["text_units"],
        "kiwi_token_count": counts["kiwi_tokens"],
        "normalization_status": {
            status: status_counts[status] for status in ("matched", "review", "unmatched")
        },
        "major_pos_and_suffix_distribution": {
            pos: pos_counts[pos] for pos in ("NNG", "MAG", "VV", "VA", "VX", "XSV", "XSA")
        },
        "derivational_candidate_count": counts["derivational_candidates"],
        "placeholder_audit": {
            "occurrences": counts["placeholder_occurrences"],
            "text_units": counts["text_units_with_placeholders"],
            "tokens_excluded": counts["placeholder_tokens_excluded"],
            "candidates_excluded": counts["placeholder_candidates_excluded"],
            "markers": ["&company&", "&brand&", "&name&", "company", "brand", "name"],
            "examples": placeholder_examples,
        },
        "non_word_audit": {
            "html_tag_occurrences": counts["html_tag_occurrences"],
            "text_units_with_html_tags": counts["text_units_with_html_tags"],
            "text_units_without_hangul": counts["text_units_without_hangul"],
            "policy": "Measured only; no broad English-word filter applied.",
        },
        "schema_paths": dict(sorted(schema_paths.items())),
        "schema_issues": [],
        "performance": {
            "elapsed_seconds": elapsed,
            "text_units_per_second": counts["text_units"] / elapsed,
            "tracemalloc_peak_mib": peak_bytes / (1024 * 1024),
            "memory_note": "Python allocation peak; excludes native Kiwi model memory.",
        },
        "normalization_examples": normalization_examples,
        "derivational_candidate_examples": candidate_examples,
    }


def main() -> int:
    args = parse_args()
    if min(args.newspaper_limit, args.dialogue_limit, args.online_limit_per_subtype) < 1:
        raise SystemExit("All pilot limits must be at least 1.")
    from kiwipiepy import Kiwi
    from kiwipiepy import __version__ as kiwi_version

    game_words = load_vocabulary(args.vocabulary)
    answer_words = load_answer_candidates(args.answer_candidates)
    kiwi = Kiwi()
    specifications: tuple[tuple[RawSource, str, int], ...] = (
        ("newspaper", "NIKL_NEWSPAPER_2025_v1.0.zip", args.newspaper_limit),
        ("dialogue", "NIKL_DIALOGUE_2025_v1.0.zip", args.dialogue_limit),
        (
            "online_ebrw",
            "NIKL_Online_Posting_Materials_Corpus_2025_v1.0.zip",
            args.online_limit_per_subtype,
        ),
        (
            "online_esrw",
            "NIKL_Online_Posting_Materials_Corpus_2025_v1.0.zip",
            args.online_limit_per_subtype,
        ),
    )
    results = {
        source: run_source(
            zip_path=args.corpus_root / filename,
            source=source,
            limit=limit,
            kiwi=kiwi,
            game_words=game_words,
            answer_words=answer_words,
        )
        for source, filename, limit in specifications
    }
    report = {
        "schema_version": "1.0",
        "kiwipiepy_version": kiwi_version,
        "scope": {
            "full_corpus_processed": False,
            "newspaper_text_units": args.newspaper_limit,
            "dialogue_utterances": args.dialogue_limit,
            "online_ebrw_units": args.online_limit_per_subtype,
            "online_esrw_units": args.online_limit_per_subtype,
        },
        "frequency_policy": {
            "derivational_candidates_added_to_frequency": False,
            "candidate_lane": "review_only",
            "placeholder_tokens_added_to_frequency": False,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
