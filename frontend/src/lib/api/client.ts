/**
 * Thin fetch wrapper for talking to the backend.
 *
 * All API calls go through here so URL construction, JSON handling and error
 * shaping live in one place. UI components must NOT call `fetch` directly.
 */

import type { ApiErrorBody } from "@/types/api";
import { API_BASE_URL } from "./config";

/** Error thrown for any non-2xx response, carrying the parsed error envelope. */
export class ApiError extends Error {
  readonly status: number;
  readonly body?: ApiErrorBody;

  constructor(status: number, body?: ApiErrorBody) {
    super(body?.message ?? `Request failed with status ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

export async function apiFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (!response.ok) {
    let body: ApiErrorBody | undefined;
    try {
      body = (await response.json()) as ApiErrorBody;
    } catch {
      // Non-JSON error body — fall back to the status-based message.
    }
    throw new ApiError(response.status, body);
  }

  return (await response.json()) as T;
}
