/**
 * TypeScript mirror of the backend wire contract.
 *
 * Field names are camelCase to match the API (see ../../../docs/API_SPEC.md).
 * Keep this file in sync with `backend/app/schemas/*` whenever the contract
 * changes — the two are the shared boundary between frontend and backend.
 */

export interface HealthResponse {
  status: "ok";
}

/** Response of the dev-only `POST /api/dev/similarity` scaffolding endpoint. */
export interface SimilarityResponse {
  first: string;
  second: string;
  similarity: number;
  provider: string;
}

/** Standard error envelope returned for handled API errors. */
export interface ApiErrorBody {
  code: string;
  message: string;
  details?: unknown | null;
}
