/** API endpoint configuration, sourced from public environment variables. */

// Fallbacks keep local dev working even without a .env.local file.
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const WS_BASE_URL =
  process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000";
