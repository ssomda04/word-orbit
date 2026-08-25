"""Single-player game use cases.

Orchestration only: a ``GameRepository`` for storage, an ``AnswerSelector`` for
the hidden word, and a ``GuessScorer`` for what a guess is worth — all injected,
so tests can substitute fakes. Rules live in ``app.domain.game``; HTTP shaping
lives in the router. This class knows nothing about status codes or response
models, and nothing about embeddings, vocabularies, or stored artifacts: it asks
the scorer one question and records the answer.

The answer word is read here only to score a guess, and is never logged.
"""

from datetime import UTC, datetime

from app.core.errors import GameAlreadyFinishedError, GameNotFoundError
from app.domain.game import Game, Guess, normalize_word
from app.domain.words import AnswerSelector
from app.services.game.repository import GameRepository
from app.services.scoring import GuessScorer


class GameService:
    """Create games, score guesses, and read game state."""

    def __init__(
        self,
        repository: GameRepository,
        scorer: GuessScorer,
        answer_selector: AnswerSelector,
    ) -> None:
        self._repository = repository
        self._scorer = scorer
        self._select_answer = answer_selector

    def create_game(self) -> Game:
        """Start a game with a server-chosen answer."""
        game = Game(
            id=self._repository.next_game_id(),
            # Normalized on the way in so a guess can never fail to match the
            # answer purely because of Unicode form or stray whitespace.
            answer=normalize_word(self._select_answer()),
            created_at=self._now(),
        )
        self._repository.add(game)
        return game

    def get_game(self, game_id: str) -> Game:
        """Return the game.

        Raises:
            GameNotFoundError: no game with that id exists.
        """
        game = self._repository.get(game_id)
        if game is None:
            raise GameNotFoundError()
        return game

    def submit_guess(self, game_id: str, raw_word: str) -> Guess:
        """Score a guess and record it.

        Re-submitting a word already guessed is idempotent: the stored result is
        returned unchanged, no new guess is created, and ``guessCount`` does not
        grow (docs/API_SPEC.md).

        Raises:
            GameNotFoundError: no game with that id exists.
            GameAlreadyFinishedError: the game has already ended.
            InvalidWordError: the word violates a guess rule.
        """
        game = self.get_game(game_id)
        # Checked before normalization and the duplicate lookup: a finished game
        # rejects *every* guess, including a replay of one already made.
        # Idempotency applies only while the game is playing.
        if game.is_finished:
            raise GameAlreadyFinishedError()

        word = normalize_word(raw_word)

        existing = game.find_guess(word)
        if existing is not None:
            return existing

        # Scored once, when the guess is first recorded. A replayed guess returns
        # the stored result above, so neither value changes after the fact even
        # if the data behind the scorer were swapped.
        score = self._scorer.score(game.answer, word)
        guess = game.record_guess(
            guess_id=self._repository.next_guess_id(game),
            word=word,
            similarity=score.similarity,
            created_at=self._now(),
            rank=score.rank,
        )
        self._repository.save(game)
        return guess

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)
