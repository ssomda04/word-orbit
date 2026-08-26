"""Adapt Kiwi tokens to the provenance-preserving Modu normalization policy."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol

from contextle_eval.modu_normalization import (
    NormalizationStatus,
    normalize_morpheme,
)

DERIVATIONAL_SUFFIX_POS = frozenset({"XSV", "XSA"})
DERIVATIONAL_BASE_POS = frozenset({"NNG", "NNP", "NNB", "XR", "SL", "SH"})
MAJOR_CONTENT_POS = frozenset({"NNG", "VV", "VA", "MAG"})


class KiwiAdapterError(ValueError):
    """Raised when Kiwi token provenance is incomplete or inconsistent."""


class KiwiTokenLike(Protocol):
    form: str
    tag: str
    start: int
    len: int
    word_position: int
    sent_position: int


@dataclass(frozen=True, slots=True)
class KiwiNormalizationRecord:
    """One Kiwi token and its shared-contract normalization decision."""

    source_text_id: str
    source_text: str
    token_index: int
    sentence_index: int
    eojeol_index: int
    token_start: int
    token_end: int
    eojeol_start: int
    eojeol_end: int
    eojeol_surface: str
    source_morpheme: str
    source_pos: str
    kiwi_tag: str
    canonical_form: str
    status: NormalizationStatus
    normalization_reason: str
    in_game_vocabulary: bool
    in_answer_candidates: bool
    eojeol_has_derivational_suffix: bool
    derivational_candidate: str


@dataclass(frozen=True, slots=True)
class DerivationalCandidateRecord:
    """Audit-only noun-base plus XSV/XSA predicate with full provenance."""

    source_text_id: str
    source_text: str
    sentence_index: int
    eojeol_index: int
    eojeol_start: int
    eojeol_end: int
    eojeol_surface: str
    base_token_index: int
    base_form: str
    base_pos: str
    suffix_token_index: int
    suffix_form: str
    suffix_pos: str
    canonical_form: str
    in_game_vocabulary: bool
    in_answer_candidates: bool
    frequency_assignment: str = "review_only"


def base_kiwi_tag(tag: str) -> str:
    """Map Kiwi regularity suffixes such as ``VV-I`` to the MP base tag."""
    return tag.split("-", maxsplit=1)[0].strip()


def adapt_kiwi_tokens(
    source_text_id: str,
    source_text: str,
    tokens: Sequence[KiwiTokenLike],
    game_words: frozenset[str],
    answer_words: frozenset[str],
) -> tuple[KiwiNormalizationRecord, ...]:
    """Normalize each actual Kiwi token once; never synthesize a second frequency."""
    if not source_text_id.strip():
        raise KiwiAdapterError("source_text_id must not be blank.")
    grouped_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, token in enumerate(tokens):
        if (
            token.start < 0
            or token.len < 0
            or token.word_position < 0
            or token.sent_position < 0
        ):
            raise KiwiAdapterError("Kiwi token has an invalid offset or word position.")
        if token.start + token.len > len(source_text):
            raise KiwiAdapterError("Kiwi token extends beyond source text.")
        grouped_indices[(token.sent_position, token.word_position)].append(index)

    eojeol_spans = {
        position_key: (
            min(tokens[index].start for index in indices),
            max(tokens[index].start + tokens[index].len for index in indices),
        )
        for position_key, indices in grouped_indices.items()
    }
    derivational_groups = {
        position_key
        for position_key, indices in grouped_indices.items()
        if any(base_kiwi_tag(tokens[index].tag) in DERIVATIONAL_SUFFIX_POS for index in indices)
    }
    derivational_candidates: dict[int, str] = {}
    for indices in grouped_indices.values():
        for previous_index, current_index in pairwise(indices):
            previous = tokens[previous_index]
            current = tokens[current_index]
            if (
                base_kiwi_tag(previous.tag) in DERIVATIONAL_BASE_POS
                and base_kiwi_tag(current.tag) in DERIVATIONAL_SUFFIX_POS
            ):
                derivational_candidates[current_index] = (
                    f"{previous.form}{current.form}다"
                )

    records: list[KiwiNormalizationRecord] = []
    for index, token in enumerate(tokens):
        source_pos = base_kiwi_tag(token.tag)
        decision = normalize_morpheme(
            token.form, source_pos, game_words, answer_words
        )
        position_key = (token.sent_position, token.word_position)
        eojeol_start, eojeol_end = eojeol_spans[position_key]
        records.append(
            KiwiNormalizationRecord(
                source_text_id=source_text_id,
                source_text=source_text,
                token_index=index,
                sentence_index=token.sent_position,
                eojeol_index=token.word_position,
                token_start=token.start,
                token_end=token.start + token.len,
                eojeol_start=eojeol_start,
                eojeol_end=eojeol_end,
                eojeol_surface=source_text[eojeol_start:eojeol_end],
                source_morpheme=token.form,
                source_pos=source_pos,
                kiwi_tag=token.tag,
                canonical_form=decision.canonical_form,
                status=decision.status,
                normalization_reason=decision.reason,
                in_game_vocabulary=decision.in_game_vocabulary,
                in_answer_candidates=decision.in_answer_candidates,
                eojeol_has_derivational_suffix=(
                    position_key in derivational_groups
                ),
                derivational_candidate=derivational_candidates.get(index, ""),
            )
        )
    return tuple(records)


def lexical_frequency_records(
    records: Sequence[KiwiNormalizationRecord],
) -> tuple[KiwiNormalizationRecord, ...]:
    """Return only matched source tokens, with no derivational synthesis or merge."""
    selected = tuple(record for record in records if record.status == "matched")
    token_keys = {(record.source_text_id, record.token_index) for record in selected}
    if len(token_keys) != len(selected):
        raise KiwiAdapterError("A Kiwi token would be counted more than once.")
    return selected


def extract_derivational_candidates(
    records: Sequence[KiwiNormalizationRecord],
    game_words: frozenset[str],
    answer_words: frozenset[str],
) -> tuple[DerivationalCandidateRecord, ...]:
    """Return review-only base+suffix candidates without adding frequency rows."""
    candidates: list[DerivationalCandidateRecord] = []
    for base, suffix in pairwise(records):
        same_eojeol = (
            base.source_text_id == suffix.source_text_id
            and base.sentence_index == suffix.sentence_index
            and base.eojeol_index == suffix.eojeol_index
        )
        if not (
            same_eojeol
            and base.source_pos in DERIVATIONAL_BASE_POS
            and suffix.source_pos in DERIVATIONAL_SUFFIX_POS
        ):
            continue
        canonical = f"{base.source_morpheme}{suffix.source_morpheme}다"
        candidates.append(
            DerivationalCandidateRecord(
                source_text_id=base.source_text_id,
                source_text=base.source_text,
                sentence_index=base.sentence_index,
                eojeol_index=base.eojeol_index,
                eojeol_start=base.eojeol_start,
                eojeol_end=base.eojeol_end,
                eojeol_surface=base.eojeol_surface,
                base_token_index=base.token_index,
                base_form=base.source_morpheme,
                base_pos=base.source_pos,
                suffix_token_index=suffix.token_index,
                suffix_form=suffix.source_morpheme,
                suffix_pos=suffix.source_pos,
                canonical_form=canonical,
                in_game_vocabulary=canonical in game_words,
                in_answer_candidates=canonical in answer_words,
            )
        )
    keys = {
        (candidate.source_text_id, candidate.base_token_index, candidate.suffix_token_index)
        for candidate in candidates
    }
    if len(keys) != len(candidates):
        raise KiwiAdapterError("A derivational source occurrence was recorded twice.")
    return tuple(candidates)
