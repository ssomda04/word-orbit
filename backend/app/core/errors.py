"""Application errors that map onto the documented error envelope.

Domain and service code raises these instead of FastAPI's ``HTTPException``,
because ``HTTPException`` serializes to ``{"detail": ...}`` while the contract
(``docs/API_SPEC.md``) is ``{"code", "message", "details"}``. A single handler in
``app.main`` performs the conversion.

This module intentionally imports nothing from FastAPI so that ``app/domain``
can raise these errors and stay framework-free.
"""

from typing import Any


class AppError(Exception):
    """Base class for errors returned in the standard error envelope.

    Subclasses declare their stable ``code`` and HTTP ``status_code``; callers may
    override the human-readable message and attach ``details``.
    """

    code: str = "INTERNAL_ERROR"
    status_code: int = 500
    message: str = "Unexpected server error."

    def __init__(self, message: str | None = None, details: Any | None = None) -> None:
        self.message = message or type(self).message
        self.details = details
        super().__init__(self.message)


class GameNotFoundError(AppError):
    code = "GAME_NOT_FOUND"
    status_code = 404
    message = "요청한 게임을 찾을 수 없습니다."


class InvalidWordError(AppError):
    """The guess cannot be processed (rule violation, or out-of-vocabulary later)."""

    code = "INVALID_WORD"
    status_code = 400
    message = "입력한 단어를 처리할 수 없습니다."


class GameAlreadyFinishedError(AppError):
    code = "GAME_ALREADY_FINISHED"
    status_code = 409
    message = "이미 종료된 게임에는 추측을 제출할 수 없습니다."
