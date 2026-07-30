export function GuessForm() {
  return (
    <form className="flex flex-col gap-3 sm:flex-row">
      <label className="sr-only" htmlFor="guess-word">
        추측 단어
      </label>

      <input
        id="guess-word"
        type="text"
        placeholder="단어를 입력하세요"
        className="min-w-0 flex-1 rounded-2xl border border-slate-200 bg-slate-50 px-5 py-4 text-base text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-red-400 focus:bg-white focus:ring-4 focus:ring-red-100"
      />

      <button
        type="button"
        className="rounded-2xl bg-red-600 px-8 py-4 font-semibold text-white transition hover:bg-red-700 active:scale-[0.98]"
      >
        추측하기
      </button>
    </form>
  );
}