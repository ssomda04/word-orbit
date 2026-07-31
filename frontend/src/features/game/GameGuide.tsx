const STEPS = [
  "정답 단어와 의미적으로 관련 있다고 생각하는 단어를 입력합니다.",
  "입력한 단어와 정답 단어 사이의 의미적 유사도와 유사도 순위를 확인합니다.",
  "유사도가 높은 단어들의 공통적인 의미를 바탕으로 다음 단어를 추측합니다.",
  "3D 임베딩 공간에서 지금까지 입력한 단어 벡터와 정답 벡터의 위치 관계를 확인합니다.",
  "정답을 맞힐 때까지 단어를 반복해서 입력하며 정답 단어에 가까워집니다.",
];

export function GameGuide() {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-lg shadow-slate-200/50 sm:p-8">
      <p className="text-sm font-semibold text-red-500">HOW TO PLAY</p>

      <h2 className="mt-1 text-2xl font-bold text-slate-900">게임 방법</h2>

      <ol className="mt-6 space-y-5">
        {STEPS.map((step, index) => (
          <li key={step} className="flex gap-4">
            <span className="w-6 shrink-0 font-semibold text-red-500">
              {index + 1}
            </span>

            <p className="pt-0.5 leading-7 text-slate-600">{step}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}