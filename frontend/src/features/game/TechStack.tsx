const STACKS = [
  {
    title: "Frontend",
    description: "Next.js, React, TypeScript, Tailwind CSS",
  },
  {
    title: "Backend",
    description: "FastAPI, Python, Pydantic",
  },
  {
    title: "Machine Learning",
    description: "Word2Vec, Cosine Similarity, PCA",
  },
  {
    title: "Deployment",
    description: "Vercel, Render 또는 Railway, GitHub Actions",
  },
];

export function TechStack() {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-lg shadow-slate-200/50 sm:p-8">
      <p className="text-sm font-semibold text-red-500">TECHNOLOGY</p>

      <h2 className="mt-1 text-2xl font-bold text-slate-900">
        사용 기술 스택
      </h2>

      <div className="mt-6 divide-y divide-slate-200">
        {STACKS.map((stack) => (
          <div
            key={stack.title}
            className="grid gap-1 py-4 first:pt-0 last:pb-0 sm:grid-cols-[150px_1fr] sm:gap-5"
          >
            <h3 className="font-semibold text-slate-800">{stack.title}</h3>

            <p className="leading-7 text-slate-600">{stack.description}</p>
          </div>
        ))}
      </div>

      <p className="mt-6 text-xs leading-5 text-slate-400">
        일부 기술은 현재 도입 예정이며 실제 구현 과정에서 변경될 수
        있습니다.
      </p>
    </section>
  );
}