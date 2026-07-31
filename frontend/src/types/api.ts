/**
 * TypeScript mirror of the backend wire contract.
 *
 * Field names are camelCase to match the API (see ../../../docs/API_SPEC.md).
 * Keep this file in sync with `backend/app/schemas/*` whenever the contract
 * changes — the two are the shared boundary between frontend and backend.
 */

export type GameStatus = "playing" | "won" | "abandoned";

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

export interface ErrorResponse {
  code: string;
  message: string;
  details: Record<string, unknown> | unknown[] | null;
}