"use client";

/**
 * Minimal client component that verifies the frontend can reach the backend.
 *
 * This is scaffolding to confirm the connection during Phase 0 — not part of
 * the game UI. It demonstrates the loading / error / ok states every real API
 * call should handle. Data fetching goes through `@/lib/api`, never inline.
 */

import { useEffect, useState } from "react";
import { API_BASE_URL } from "@/lib/api/config";
import { fetchHealth } from "@/lib/api/health";

type Status =
  | { kind: "loading" }
  | { kind: "ok"; value: string }
  | { kind: "error"; message: string };

const DOT: Record<Status["kind"], string> = {
  loading: "bg-amber-400",
  ok: "bg-emerald-500",
  error: "bg-rose-500",
};

export function HealthStatus() {
  const [status, setStatus] = useState<Status>({ kind: "loading" });

  useEffect(() => {
    let active = true;
    fetchHealth()
      .then((res) => {
        if (active) setStatus({ kind: "ok", value: res.status });
      })
      .catch((err: unknown) => {
        if (active) {
          setStatus({
            kind: "error",
            message: err instanceof Error ? err.message : "Unknown error",
          });
        }
      });
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="w-full max-w-md rounded-xl border border-black/10 p-5 dark:border-white/15">
      <div className="mb-2 flex items-center gap-2">
        <span className={`h-2.5 w-2.5 rounded-full ${DOT[status.kind]}`} />
        <span className="text-sm font-medium">Backend /health</span>
      </div>

      <p className="text-sm text-zinc-600 dark:text-zinc-400">
        {status.kind === "loading" && "Checking backend connection…"}
        {status.kind === "ok" && (
          <>
            Connected — status: <code className="font-mono">{status.value}</code>
          </>
        )}
        {status.kind === "error" && (
          <>
            Cannot reach backend:{" "}
            <span className="text-rose-600 dark:text-rose-400">{status.message}</span>
          </>
        )}
      </p>

      <p className="mt-3 break-all text-xs text-zinc-400 dark:text-zinc-500">
        {API_BASE_URL}/health
      </p>
    </div>
  );
}
