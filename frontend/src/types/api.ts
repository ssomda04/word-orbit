/**
 * TypeScript mirror of the backend wire contract.
 *
 * Field names are camelCase to match the API (see ../../../docs/API_SPEC.md).
 * Keep this file in sync with `backend/app/schemas/*` whenever the contract
 * changes — the two are the shared boundary between frontend and backend.
 */

export interface HealthResponse {
  status: string;
}

export type GameStatus = "playing" | "won" | "abandoned";

/** Why a finished game ended. Derived from `GameStatus` by the server. */
export type FinishReason = "correct" | "gave_up";

export interface Coordinate {
  x: number;
  y: number;
  z: number;
}

export interface CreateGameResponse {
  gameId: string;
  status: GameStatus;
  createdAt: string;
}

export interface Guess {
  guessId: string;
  word: string;
  similarity: number;
  rank: number | null;
  isAnswer: boolean;
  coordinate: Coordinate | null;
}

export interface GameStateResponse {
  gameId: string;
  status: GameStatus;
  createdAt: string;
  guessCount: number;
  guesses: Guess[];
  answer: string | null;
}

/**
 * Response of `POST /api/games/{gameId}/give-up`.
 *
 * `answer` is non-nullable here, unlike on `GameStateResponse`: this response
 * only exists for a game that has just ended.
 */
export interface GiveUpResponse {
  gameId: string;
  status: GameStatus;
  finishReason: FinishReason;
  answer: string;
}

export interface ErrorResponse {
  code: string;
  message: string;
  details: Record<string, unknown> | unknown[] | null;
}

export type ApiErrorBody = ErrorResponse;