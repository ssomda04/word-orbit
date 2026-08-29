import type {
  CreateGameResponse,
  ErrorResponse,
  GameStateResponse,
  GiveUpResponse,
  Guess,
} from "@/types/api";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class GameApiError extends Error {
  status: number;
  code: string;
  details: Record<string, unknown> | unknown[] | null;

  constructor(status: number, error: ErrorResponse) {
    super(error.message);

    this.name = "GameApiError";
    this.status = status;
    this.code = error.code;
    this.details = error.details;
  }
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (response.ok) {
    return (await response.json()) as T;
  }

  let errorResponse: ErrorResponse;

  try {
    errorResponse = (await response.json()) as ErrorResponse;
  } catch {
    errorResponse = {
      code: "UNKNOWN_ERROR",
      message: "서버 응답을 처리하지 못했습니다.",
      details: null,
    };
  }

  throw new GameApiError(response.status, errorResponse);
}

/**
 * 백엔드 서버가 정상 실행 중인지 확인한다.
 */
export async function checkHealth(): Promise<{ status: string }> {
  const response = await fetch(`${API_URL}/health`, {
    method: "GET",
    cache: "no-store",
  });

  return parseResponse<{ status: string }>(response);
}

/**
 * 새로운 싱글플레이 게임을 생성한다.
 */
export async function createGame(): Promise<CreateGameResponse> {
  const response = await fetch(`${API_URL}/api/games`, {
    method: "POST",
    cache: "no-store",
  });

  return parseResponse<CreateGameResponse>(response);
}

/**
 * 현재 게임에 추측 단어를 제출한다.
 */
export async function submitGuess(
  gameId: string,
  word: string,
): Promise<Guess> {
  const response = await fetch(
    `${API_URL}/api/games/${gameId}/guesses`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ word }),
    },
  );

  return parseResponse<Guess>(response);
}

/**
 * 현재 게임 상태와 추측 기록을 조회한다.
 */
export async function getGameState(
  gameId: string,
): Promise<GameStateResponse> {
  const response = await fetch(`${API_URL}/api/games/${gameId}`, {
    method: "GET",
    cache: "no-store",
  });

  return parseResponse<GameStateResponse>(response);
}

/**
 * 현재 진행 중인 게임을 포기하고 정답을 공개한다.
 */
export async function giveUpGame(
  gameId: string,
): Promise<GiveUpResponse> {
  const response = await fetch(
    `${API_URL}/api/games/${gameId}/give-up`,
    {
      method: "POST",
      cache: "no-store",
    },
  );

  return parseResponse<GiveUpResponse>(response);
}
