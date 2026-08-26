#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "kiwipiepy==0.23.2",
# ]
# ///
"""Inspect small Kiwi fixtures against the Modu normalization adapter."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextle_eval.modu_baseform_analysis import (  # noqa: E402
    load_answer_candidates,
    load_vocabulary,
)
from contextle_eval.modu_kiwi_adapter import adapt_kiwi_tokens  # noqa: E402

DEFAULT_VOCABULARY = ML_ROOT / "data" / "game_words.txt"
DEFAULT_ANSWER_CANDIDATES = ML_ROOT / "data" / "answer_candidates.csv"
FIXTURES = {
    "newspaper": (
        "어제는 밥을 먹었다. 노래를 들었고 길을 걸었다. "
        "자료가 공개됐고 의견은 다양했다."
    ),
    "dialogue": (
        "날씨가 좋다. 어제는 추웠다. 그곳에 사람이 있고 문제는 없어. "
        "두 결과가 같아. 공부하지 않고 한번 해 봐."
    ),
    "online": (
        "오늘 서버가 다운됐네요. 아 진짜 좋다ㅋㅋ "
        "친구를 도와주고 음악을 좋아해요."
    ),
    "derivational": (
        "밥을 먹다. 노래를 듣다. 길을 걷다. 일을 하다. 준비가 되다. "
        "공부하다. 생각하다. 가능하다. 행복하다. 좋아하다. "
        "친구를 도와주다. 자료가 공개되다. 의견이 다양하다."
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect bounded Kiwi fixture output.")
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    parser.add_argument("--answer-candidates", type=Path, default=DEFAULT_ANSWER_CANDIDATES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from kiwipiepy import Kiwi
    from kiwipiepy import __version__ as kiwi_version

    kiwi = Kiwi()
    game_words = load_vocabulary(args.vocabulary)
    answer_words = load_answer_candidates(args.answer_candidates)
    output = {"kiwipiepy_version": kiwi_version, "fixtures": {}}
    for fixture_id, text in FIXTURES.items():
        tokens = kiwi.tokenize(text)
        records = adapt_kiwi_tokens(
            fixture_id, text, tokens, game_words, answer_words
        )
        output["fixtures"][fixture_id] = {
            "text": text,
            "raw_tokens": [
                {
                    "form": token.form,
                    "tag": token.tag,
                    "start": token.start,
                    "end": token.end,
                    "length": token.len,
                    "word_position": token.word_position,
                    "sent_position": token.sent_position,
                    "lemma": token.lemma,
                    "base_form": token.base_form,
                    "raw_form": token.raw_form,
                    "regularity": token.regularity,
                }
                for token in tokens
            ],
            "tokens": [asdict(record) for record in records],
        }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
