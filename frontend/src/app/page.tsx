import { HealthStatus } from "@/features/health/HealthStatus";

export default function Home() {
  return (
    <main className="flex flex-1 flex-col items-center justify-center gap-8 px-6 py-16">
      <div className="flex flex-col items-center gap-3 text-center">
        <h1 className="text-4xl font-semibold tracking-tight">Contextle</h1>
        <p className="max-w-md text-zinc-600 dark:text-zinc-400">
          의미 기반 단어 추론 게임 — 개발 환경 스캐폴드
        </p>
        <p className="text-sm font-medium text-emerald-600 dark:text-emerald-400">
          Frontend is running ✅
        </p>
      </div>

      <HealthStatus />

      <p className="max-w-md text-center text-xs text-zinc-400 dark:text-zinc-500">
        This page only verifies the frontend&nbsp;↔&nbsp;backend connection. The
        game UI is not implemented yet — see <code>docs/ROADMAP.md</code>.
      </p>
    </main>
  );
}
