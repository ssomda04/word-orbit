import { EmbeddingSpace } from "./EmbeddingSpace";
import { GameGuide } from "./GameGuide";
import { GuessForm } from "./GuessForm";
import { GuessHistory } from "./GuessHistory";
import { ProjectIntroduction } from "./ProjectIntroduction";
import { TechStack } from "./TechStack";

export function GameView() {
  return (
    <div className="flex flex-col gap-8">
      <section className="rounded-[32px] border border-slate-200 bg-white p-6 shadow-xl shadow-slate-300/40 sm:p-9">
        <div className="text-center">
          <span className="inline-flex rounded-full bg-red-50 px-4 py-2 text-sm font-semibold text-red-600">
            오늘의 단어
          </span>

          <h2 className="mt-5 text-3xl font-bold text-slate-950">
            정답 단어를 추론해 보세요
          </h2>

          <p className="mx-auto mt-3 max-w-2xl leading-7 text-slate-500">
            단어를 입력하면 정답 단어와의 의미적 유사도와 순위를 확인할 수
            있습니다.
          </p>
        </div>

        <div className="mt-8">
          <GuessForm />
        </div>

        <div className="mt-6 grid gap-4 sm:grid-cols-2">
          <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
            <p className="text-sm text-slate-500">시도 횟수</p>
            <p className="mt-1 text-2xl font-bold text-slate-900">0회</p>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
            <p className="text-sm text-slate-500">현재 최고 유사도</p>
            <p className="mt-1 text-2xl font-bold text-red-500">-</p>
          </div>
        </div>
      </section>

      <section className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
        <GuessHistory />
        <EmbeddingSpace />
      </section>

      <GameGuide />
      <ProjectIntroduction />
      <TechStack />
    </div>
  );
}