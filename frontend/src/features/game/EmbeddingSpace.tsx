import { EmbeddingScene } from "./EmbeddingScene";

export function EmbeddingSpace() {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-lg shadow-slate-200/50">
      <div>
        <p className="text-sm font-semibold text-red-500">VECTOR SPACE</p>

        <h2 className="mt-1 text-2xl font-bold text-slate-900">
          임베딩 공간 3D 시각화
        </h2>

        <p className="mt-3 text-sm leading-6 text-slate-500">
          마우스로 공간을 회전하거나 확대·축소할 수 있습니다. 추후 사용자가
          입력한 단어 벡터와 정답 벡터가 이 좌표 공간에 표시됩니다.
        </p>
      </div>

      <div className="mt-6 h-96 overflow-hidden rounded-2xl border border-slate-200 bg-slate-50">
        <EmbeddingScene />
      </div>

      <div className="mt-4 grid gap-2 text-sm text-slate-500 sm:grid-cols-3">
        <p>
          <span className="font-semibold text-slate-700">회전:</span> 마우스 왼쪽
          버튼을 누른 채 이동
        </p>

        <p>
          <span className="font-semibold text-slate-700">확대·축소:</span> 마우스
          휠
        </p>

        <p>
          <span className="font-semibold text-slate-700">이동:</span> 마우스 오른쪽
          버튼을 누른 채 이동
        </p>
      </div>

      <div className="mt-5 flex flex-wrap gap-4 text-sm text-slate-500">
        <span className="flex items-center gap-2">
          <span className="h-2.5 w-5 rounded-full bg-red-500" />
          X축
        </span>

        <span className="flex items-center gap-2">
          <span className="h-2.5 w-5 rounded-full bg-green-500" />
          Y축
        </span>

        <span className="flex items-center gap-2">
          <span className="h-2.5 w-5 rounded-full bg-blue-500" />
          Z축
        </span>
      </div>

      <p className="mt-4 text-xs leading-5 text-slate-400">
        ※ 현재는 좌표축과 기준 격자만 표시합니다. 실제 단어 좌표는 백엔드의
        차원 축소 결과를 연결한 뒤 추가할 예정입니다.
      </p>
    </section>
  );
}