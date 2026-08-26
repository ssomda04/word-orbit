#!/usr/bin/env python3
"""Run a bounded MP-versus-Kiwi compatibility audit on NXMP and SXMP."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path
from typing import Any

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextle_eval.modu_corpus import (
    MorphemeRecord,
    ParsedSentence,
    SourceSubtype,
    iter_mp_sentences,
)
from contextle_eval.modu_kiwi_adapter import base_kiwi_tag

CONTENT_POS = frozenset({"NNG", "MAG", "VV", "VA"})
PREDICATE_POS = frozenset({"VV", "VA", "VX"})
DERIVATIONAL_POS = frozenset({"XSV", "XSA"})
FOCUS_FORMS = (
    "있다",
    "없다",
    "같다",
    "하다",
    "되다",
    "보다",
    "공부하다",
    "생각하다",
    "가능하다",
    "행복하다",
    "공개되다",
    "좋아하다",
    "도와주다",
)
MAX_EXAMPLES = 20


@dataclass(frozen=True, slots=True)
class Unit:
    form: str
    pos: str
    raw_pos: str

    @property
    def canonical(self) -> str:
        return f"{self.form}다" if self.pos in PREDICATE_POS else self.form


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--sentences", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=ML_ROOT / "data/modu_frequency/kiwi_mp_compatibility.json",
    )
    return parser.parse_args()


def mp_groups(sentence: ParsedSentence) -> list[tuple[str, list[Unit]]]:
    grouped: dict[int, list[MorphemeRecord]] = defaultdict(list)
    forms: dict[int, str] = {}
    for record in sentence.records:
        if record.word_id is not None and record.word_form is not None:
            grouped[record.word_id].append(record)
            forms[record.word_id] = record.word_form
    return [
        (
            forms[word_id],
            [Unit(record.morpheme, record.pos, record.pos) for record in records],
        )
        for word_id, records in grouped.items()
    ]


def kiwi_groups(text: str, tokens: list[Any]) -> list[tuple[str, list[Unit]]]:
    spans = list(re.finditer(r"\S+", text))
    grouped: dict[int, list[Any]] = defaultdict(list)
    for token in tokens:
        for index, span in enumerate(spans):
            if span.start() <= token.start < span.end():
                grouped[index].append(token)
                break
    output: list[tuple[str, list[Unit]]] = []
    for index, span in enumerate(spans):
        group = grouped[index]
        output.append(
            (
                span.group(),
                [Unit(token.form, base_kiwi_tag(token.tag), token.tag) for token in group],
            )
        )
    return output


def content(units: list[Unit]) -> list[Unit]:
    return [unit for unit in units if unit.pos in CONTENT_POS]


def lexical_counter(units: list[Unit], *, include_derived: bool) -> Counter[str]:
    values = Counter(unit.canonical for unit in units if unit.pos in CONTENT_POS)
    if include_derived:
        for left, right in pairwise(units):
            if left.pos in {"NNG", "NNP", "NNB", "XR", "SL", "SH"} and right.pos in (
                DERIVATIONAL_POS
            ):
                values[f"{left.form}{right.form}다"] += 1
    return values


def serialize_units(units: list[Unit]) -> list[dict[str, str]]:
    return [asdict(unit) | {"canonical": unit.canonical} for unit in units]


def append_example(target: list[dict[str, Any]], value: dict[str, Any]) -> None:
    if len(target) < MAX_EXAMPLES:
        target.append(value)


def audit_subtype(zip_path: Path, subtype: SourceSubtype, limit: int, kiwi: Any) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    pos_counts = {pos: Counter() for pos in CONTENT_POS}
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    focus: dict[str, Counter[str]] = {form: Counter() for form in FOCUS_FORMS}
    irregular = Counter()

    sentences = list(
        iter_mp_sentences(zip_path, source_subtype=subtype, limit_sentences=limit)
    )
    for sentence in sentences:
        tokens = list(kiwi.tokenize(sentence.sentence_form))
        mp_eojeols = mp_groups(sentence)
        kiwi_eojeols = kiwi_groups(sentence.sentence_form, tokens)
        counts["mp_tokens"] += len(sentence.records)
        counts["kiwi_tokens"] += len(tokens)
        counts["mp_eojeols"] += len(mp_eojeols)
        counts["kiwi_eojeols"] += len(kiwi_eojeols)
        for token in tokens:
            tag = str(token.tag)
            if tag in {"VV-I", "VA-I"}:
                irregular[f"{tag}->{base_kiwi_tag(tag)}"] += 1

        if [surface for surface, _ in mp_eojeols] != [
            surface for surface, _ in kiwi_eojeols
        ]:
            append_example(
                examples["tokenization"],
                {
                    "sentence_id": sentence.sentence_id,
                    "text": sentence.sentence_form,
                    "mp_eojeol_count": len(mp_eojeols),
                    "kiwi_eojeol_count": len(kiwi_eojeols),
                },
            )

        matcher = SequenceMatcher(
            a=[surface for surface, _ in mp_eojeols],
            b=[surface for surface, _ in kiwi_eojeols],
            autojunk=False,
        )
        aligned_pairs: list[tuple[int, int]] = []
        for tag, mp_start, mp_end, kiwi_start, kiwi_end in matcher.get_opcodes():
            if tag == "equal":
                aligned_pairs.extend(
                    zip(
                        range(mp_start, mp_end),
                        range(kiwi_start, kiwi_end),
                        strict=True,
                    )
                )
            else:
                counts["unaligned_mp_eojeols"] += mp_end - mp_start
                counts["unaligned_kiwi_eojeols"] += kiwi_end - kiwi_start
                append_example(
                    examples["surface_mismatch"],
                    {
                        "sentence_id": sentence.sentence_id,
                        "mp_surfaces": [
                            surface for surface, _ in mp_eojeols[mp_start:mp_end]
                        ],
                        "kiwi_surfaces": [
                            surface for surface, _ in kiwi_eojeols[kiwi_start:kiwi_end]
                        ],
                    },
                )

        for mp_index, kiwi_index in aligned_pairs:
            mp_surface, mp_units = mp_eojeols[mp_index]
            kiwi_surface, kiwi_units = kiwi_eojeols[kiwi_index]
            assert mp_surface == kiwi_surface
            counts["surface_aligned_eojeols"] += 1
            if [unit.form for unit in mp_units] == [unit.form for unit in kiwi_units]:
                counts["exact_full_tokenization_eojeols"] += 1
            else:
                counts["different_tokenization_eojeols"] += 1
            mp_content = content(mp_units)
            kiwi_content = content(kiwi_units)
            mp_lexical = lexical_counter(mp_units, include_derived=False)
            kiwi_direct_lexical = lexical_counter(kiwi_units, include_derived=False)
            kiwi_lexical = lexical_counter(kiwi_units, include_derived=True)
            counts["canonical_mp"] += sum(mp_lexical.values())
            counts["canonical_direct_matches"] += sum(
                (mp_lexical & kiwi_direct_lexical).values()
            )
            canonical_matches = sum((mp_lexical & kiwi_lexical).values())
            counts["canonical_matches"] += canonical_matches

            for form in FOCUS_FORMS:
                mp_has = mp_lexical[form] > 0
                kiwi_has = kiwi_lexical[form] > 0
                if mp_has:
                    focus[form]["mp_occurrences"] += mp_lexical[form]
                    focus[form]["kiwi_canonical_matches"] += min(
                        mp_lexical[form], kiwi_lexical[form]
                    )
                    if kiwi_has and not any(unit.canonical == form for unit in kiwi_content):
                        focus[form]["structural_matches"] += 1

            if [(unit.form, unit.pos) for unit in mp_content] == [
                (unit.form, unit.pos) for unit in kiwi_content
            ]:
                counts["exact_content_eojeols"] += 1
                for unit in mp_content:
                    pos_counts[unit.pos]["compared"] += 1
                    pos_counts[unit.pos]["agreed"] += 1
            elif [unit.canonical for unit in mp_content] == [
                unit.canonical for unit in kiwi_content
            ] and len(mp_content) == len(kiwi_content):
                counts["canonical_sequence_eojeols"] += 1
                for mp_unit, kiwi_unit in zip(mp_content, kiwi_content, strict=True):
                    pos_counts[mp_unit.pos]["compared"] += 1
                    if mp_unit.pos == kiwi_unit.pos:
                        pos_counts[mp_unit.pos]["agreed"] += 1
                    else:
                        pos_counts[mp_unit.pos][f"kiwi_{kiwi_unit.pos}"] += 1
                        append_example(
                            examples["same_canonical_pos_mismatch"],
                            {
                                "sentence_id": sentence.sentence_id,
                                "surface": mp_surface,
                                "canonical": mp_unit.canonical,
                                "mp_pos": mp_unit.pos,
                                "kiwi_pos": kiwi_unit.pos,
                            },
                        )
            else:
                append_example(
                    examples["content_mismatch"],
                    {
                        "sentence_id": sentence.sentence_id,
                        "surface": mp_surface,
                        "mp": serialize_units(mp_units),
                        "kiwi": serialize_units(kiwi_units),
                        "canonical_overlap": sorted((mp_lexical & kiwi_lexical).elements()),
                    },
                )

            for mp_unit in mp_units:
                if mp_unit.pos in PREDICATE_POS:
                    exact = [unit for unit in kiwi_units if unit.canonical == mp_unit.canonical]
                    if exact and all(unit.pos != mp_unit.pos for unit in exact):
                        counts["predicate_pos_disagreements"] += 1
                        counts[f"predicate_{mp_unit.pos}_to_{exact[0].pos}"] += 1
                        append_example(
                            examples["predicate_pos_disagreement"],
                            {
                                "sentence_id": sentence.sentence_id,
                                "surface": mp_surface,
                                "canonical": mp_unit.canonical,
                                "mp_pos": mp_unit.pos,
                                "kiwi_pos": sorted({unit.pos for unit in exact}),
                            },
                        )
                    derived = []
                    for left, right in pairwise(kiwi_units):
                        candidate = f"{left.form}{right.form}다"
                        if right.pos in DERIVATIONAL_POS and candidate == mp_unit.canonical:
                            derived.append(f"{left.form}/{left.pos}+{right.form}/{right.pos}")
                    if derived:
                        counts["derivational_splits"] += 1
                        suffix_pos = derived[0].split("/")[-1]
                        counts[f"derivational_{suffix_pos}"] += 1
                        append_example(
                            examples["derivational_split"],
                            {
                                "sentence_id": sentence.sentence_id,
                                "surface": mp_surface,
                                "mp": f"{mp_unit.form}/{mp_unit.pos}",
                                "kiwi": derived,
                                "canonical": mp_unit.canonical,
                            },
                        )

    canonical_total = counts["canonical_mp"]
    pos_output: dict[str, dict[str, Any]] = {}
    for pos in sorted(CONTENT_POS):
        values = pos_counts[pos]
        compared = values["compared"]
        pos_output[pos] = dict(sorted(values.items())) | {
            "agreement_rate": values["agreed"] / compared if compared else None
        }
    return {
        "sentences_analyzed": len(sentences),
        "mp_token_count": counts["mp_tokens"],
        "kiwi_token_count": counts["kiwi_tokens"],
        "alignment": {
            "mp_eojeols": counts["mp_eojeols"],
            "kiwi_eojeols": counts["kiwi_eojeols"],
            "surface_aligned_eojeols": counts["surface_aligned_eojeols"],
            "unaligned_mp_eojeols": counts["unaligned_mp_eojeols"],
            "unaligned_kiwi_eojeols": counts["unaligned_kiwi_eojeols"],
            "exact_content_eojeols": counts["exact_content_eojeols"],
            "canonical_sequence_eojeols": counts["canonical_sequence_eojeols"],
            "exact_full_tokenization_eojeols": counts[
                "exact_full_tokenization_eojeols"
            ],
            "different_tokenization_eojeols": counts[
                "different_tokenization_eojeols"
            ],
        },
        "content_pos_agreement": pos_output,
        "canonical_form_agreement": {
            "mp_content_occurrences": canonical_total,
            "direct_contract_matched_occurrences": counts[
                "canonical_direct_matches"
            ],
            "direct_contract_recall": (
                counts["canonical_direct_matches"] / canonical_total
                if canonical_total
                else None
            ),
            "structural_assisted_matched_occurrences": counts["canonical_matches"],
            "structural_assisted_recall": (
                counts["canonical_matches"] / canonical_total if canonical_total else None
            ),
            "definition": (
                "multiset recall of MP NNG/MAG/VV/VA canonical forms in the same "
                "surface-aligned eojeol; Kiwi noun+XSV/XSA candidates are audit-only"
            ),
        },
        "irregular_tag_mapping": dict(sorted(irregular.items())),
        "structural_disagreement_counts": {
            "predicate_pos_disagreements": counts["predicate_pos_disagreements"],
            "predicate_transitions": {
                key.removeprefix("predicate_"): value
                for key, value in sorted(counts.items())
                if key.startswith("predicate_") and key != "predicate_pos_disagreements"
            },
            "derivational_splits": counts["derivational_splits"],
            "XSV_splits": counts["derivational_XSV"],
            "XSA_splits": counts["derivational_XSA"],
        },
        "focus_forms": {
            form: dict(sorted(values.items())) for form, values in focus.items() if values
        },
        "examples": dict(sorted(examples.items())),
    }


def main() -> int:
    args = parse_args()
    if args.sentences < 1:
        raise SystemExit("--sentences must be at least 1")
    from kiwipiepy import Kiwi
    from kiwipiepy import __version__ as kiwi_version

    kiwi = Kiwi()
    results = {
        subtype: audit_subtype(args.zip_path, subtype, args.sentences, kiwi)
        for subtype in ("NXMP", "SXMP")
    }
    report = {
        "schema_version": "1.0",
        "bounded_scope": {
            "source": str(args.zip_path),
            "sentence_limit_per_subtype": args.sentences,
            "subtypes": ["NXMP", "SXMP"],
            "full_corpus_processed": False,
        },
        "kiwipiepy_version": kiwi_version,
        "method": {
            "alignment_policy": (
                "Compare by ordered eojeol only when MP and Kiwi surfaces are equal; "
                "direct POS comparison requires identical canonical content-token sequences."
            ),
            "irregular_policy": "VV-I/VA-I are mapped to VV/VA with base_kiwi_tag.",
            "normalization_contract": "NNG/MAG raw; VV/VA plus 다; other POS excluded.",
        },
        "results": results,
        "normalization_impact": {
            "derivational_candidates_are_not_frequency_records": True,
            "decision": "B. usable after a small policy reinforcement",
            "recommended_reinforcement": (
                "Keep direct token normalization unchanged, but retain noun+XSV/XSA "
                "predicate candidates in a separate review/provenance lane before a raw "
                "three-corpus pilot; do not silently count both noun and synthesized predicate."
            ),
            "reason": (
                "Directly alignable NNG/MAG/VV/VA POS is highly consistent, while "
                "productive predicates are often represented as noun+XSV/XSA by Kiwi."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} ({args.sentences} sentences per subtype)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
