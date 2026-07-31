export function GuessHistory() {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-lg shadow-slate-200/50">
      <div className="flex items-end justify-between gap-4">
        <div>
          <p className="text-sm font-semibold text-red-500">ORBIT LOG</p>
          <h2 className="mt-1 text-2xl font-bold text-slate-900">
            의미 탐색 기록
          </h2>
        </div>

        <span className="rounded-full bg-slate-100 px-3 py-1 text-sm text-slate-600">
          총 0개
        </span>
      </div>

      <p className="mt-3 text-sm text-slate-500">
        유사도가 높을수록 빨간색에 가깝게 표시됩니다.
      </p>

      <div className="mt-5 flex flex-wrap gap-4 text-xs text-slate-500">
        <span className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full bg-blue-400" />
          의미적으로 멀리 있음
        </span>

        <span className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full bg-slate-300" />
          중간
        </span>

        <span className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full bg-red-500" />
          의미적으로 가까움
        </span>
      </div>

      <div className="mt-6 flex min-h-72 items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-5 text-center">
        <div>
          <p className="font-medium text-slate-600">
            아직 입력한 단어가 없습니다.
          </p>

          <p className="mt-2 text-sm leading-6 text-slate-400">
            단어를 입력하면 유사도와 순위가 이곳에 표시됩니다.
          </p>
        </div>
      </div>
    </section>
  );
}