#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "ijson>=3.4,<4",
# ]
# ///
"""Inspect a bounded NXMP sample directly inside a NIKL Modu ZIP."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextle_eval.modu_corpus import ModuCorpusError, iter_nxmp_sentences

DEFAULT_LIMIT = 100
DEFAULT_SAMPLE_RECORDS = 50


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream and validate a bounded NXMP sample without extracting its ZIP."
    )
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--limit-sentences", type=_positive_int, default=DEFAULT_LIMIT)
    parser.add_argument("--sample-records", type=_positive_int, default=DEFAULT_SAMPLE_RECORDS)
    return parser.parse_args(argv)


def inspect(zip_path: Path, limit_sentences: int, sample_size: int) -> dict[str, object]:
    pos_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    samples: list[dict[str, object]] = []
    sentence_count = word_count = mp_count = 0
    corpus_id = ""

    for sentence in iter_nxmp_sentences(zip_path, limit_sentences=limit_sentences):
        corpus_id = sentence.corpus_id
        sentence_count += 1
        word_count += sentence.word_count
        mp_count += len(sentence.records)
        pos_counts.update(record.pos for record in sentence.records if record.pos)
        issue_counts.update(issue.code for issue in sentence.issues)
        remaining = sample_size - len(samples)
        if remaining > 0:
            samples.extend(asdict(record) for record in sentence.records[:remaining])

    return {
        "corpus_id": corpus_id,
        "source_subtype": "NXMP",
        "sentence_limit": limit_sentences,
        "parsed_sentences": sentence_count,
        "parsed_words": word_count,
        "parsed_mp": mp_count,
        "pos_distribution": dict(sorted(pos_counts.items())),
        "orphan_word_id": issue_counts["orphan_word_id"],
        "missing_word_id": issue_counts["missing_word_id"],
        "position_anomalies": issue_counts["position_order"],
        "empty_morpheme": issue_counts["empty_morpheme"],
        "empty_pos": issue_counts["empty_pos"],
        "duplicate_mp_id": issue_counts["duplicate_mp_id"],
        "sample_records": samples,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = inspect(args.zip_path, args.limit_sentences, args.sample_records)
    except (ModuCorpusError, OSError) as exc:
        print(f"NXMP inspection failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
