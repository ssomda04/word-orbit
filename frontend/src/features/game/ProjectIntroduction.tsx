export function ProjectIntroduction() {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-lg shadow-slate-200/50 sm:p-8">
      <p className="text-sm font-semibold text-red-500">ABOUT PROJECT</p>

      <h2 className="mt-1 text-2xl font-bold text-slate-900">
        프로젝트 소개
      </h2>

      <p className="mt-6 leading-8 text-slate-600">
        본 프로젝트는 Word2Vec과 Word Embedding의 원리를 실제 웹서비스에
        적용하기 위해 제작되었습니다. 단어를 고차원 벡터로 표현하고,
        코사인 유사도를 이용하여 단어 사이의 의미적 거리를 계산합니다.
        단순한 단어 맞히기 게임을 넘어 사용자가 추측한 단어와 정답 단어의
        관계를 임베딩 공간에서 시각적으로 확인함으로써 벡터 공간의 개념을
        직관적으로 이해할 수 있도록 하는 것을 목표로 합니다.
      </p>
    </section>
  );
}