/** Health endpoint client. */

import type { HealthResponse } from "@/types/api";
import { apiFetch } from "./client";

export function fetchHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/health");
}
