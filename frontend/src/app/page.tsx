"use client";

import { FormEvent, useMemo, useState } from "react";
import styles from "./page.module.css";

type SpectralType = "M" | "K" | "G" | "F" | "A" | "B" | "O";

type Guess = {
  id: number;
  word: string;
  similarity: number;
  rank: number;
  spectralType: SpectralType;
  x: number;
  y: number;
};

const initialGuesses: Guess[] = [
  {
    id: 1,
    word: "사과",
    similarity: 74.2,
    rank: 128,
    spectralType: "G",
    x: 23,
    y: 34,
  },
  {
    id: 2,
    word: "과일",
    similarity: 81.7,
    rank: 42,
    spectralType: "F",
    x: 38,
    y: 24,
  },
  {
    id: 3,
    word: "복숭아",
    similarity: 91.3,
    rank: 7,
    spectralType: "B",
    x: 45,
    y: 48,
  },
];

const spectralTypes: SpectralType[] = ["M", "K", "G", "F", "A", "B", "O"];

function getSpectralType(similarity: number): SpectralType {
  if (similarity >= 95) return "O";
  if (similarity >= 88) return "B";
  if (similarity >= 80) return "A";
  if (similarity >= 70) return "F";
  if (similarity >= 55) return "G";
  if (similarity >= 35) return "K";
  return "M";
}

function createMockResult(word: string, id: number): Guess {
  const similarity = Number((30 + Math.random() * 68).toFixed(1));

  return {
    id,
    word,
    similarity,
    rank: Math.max(1, Math.floor((100 - similarity) * 18)),
    spectralType: getSpectralType(similarity),
    x: 12 + Math.random() * 75,
    y: 14 + Math.random() * 68,
  };
}

export default function Home() {
  const [input, setInput] = useState("");
  const [guesses, setGuesses] = useState<Guess[]>(initialGuesses);
  const [selectedGuess, setSelectedGuess] = useState<Guess | null>(
    initialGuesses[2]
  );

  const bestGuess = useMemo(() => {
    return [...guesses].sort((a, b) => b.similarity - a.similarity)[0];
  }, [guesses]);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedWord = input.trim();

    if (!trimmedWord) {
      return;
    }

    const newGuess = createMockResult(trimmedWord, Date.now());

    setGuesses((previous) => [...previous, newGuess]);
    setSelectedGuess(newGuess);
    setInput("");
  }

  return (
    <main className={styles.page}>
      <div className={styles.backgroundStars} />

      <section className={styles.app}>
        <header className={styles.header}>
          <div>
            <div className={styles.logoRow}>
              <span className={styles.logoStar}>✦</span>
              <h1>WORD ORBIT</h1>
            </div>

            <p>Find the hidden word in semantic space</p>
          </div>

          <nav className={styles.navigation} aria-label="주요 메뉴">
            <button type="button">도움말</button>
            <button type="button">프로젝트</button>
          </nav>
        </header>

        <div className={styles.mainContent}>
          <section className={styles.spacePanel}>
            <div className={styles.spaceHeader}>
              <div>
                <span className={styles.eyebrow}>SEMANTIC UNIVERSE</span>
                <h2>단어 임베딩 우주</h2>
              </div>

              <div className={styles.spaceStatus}>
                <span className={styles.statusDot} />
                탐색 중
              </div>
            </div>

            <div className={styles.universe}>
              <div className={styles.orbitLarge} />
              <div className={styles.orbitSmall} />

              <svg
                className={styles.constellationLines}
                viewBox="0 0 100 100"
                preserveAspectRatio="none"
                aria-hidden="true"
              >
                {guesses.slice(1).map((guess, index) => {
                  const previousGuess = guesses[index];

                  return (
                    <line
                      key={`${previousGuess.id}-${guess.id}`}
                      x1={previousGuess.x}
                      y1={previousGuess.y}
                      x2={guess.x}
                      y2={guess.y}
                    />
                  );
                })}
              </svg>

              <button
                type="button"
                className={`${styles.answerStar} ${styles.starO}`}
                aria-label="정답 단어"
              >
                <span className={styles.answerCore} />
                <span className={styles.answerPulse} />
              </button>

              {guesses.map((guess) => (
                <button
                  type="button"
                  key={guess.id}
                  className={[
                    styles.wordStar,
                    styles[`star${guess.spectralType}`],
                    selectedGuess?.id === guess.id
                      ? styles.selectedStar
                      : "",
                    bestGuess?.id === guess.id ? styles.bestStar : "",
                  ].join(" ")}
                  style={{
                    left: `${guess.x}%`,
                    top: `${guess.y}%`,
                  }}
                  onClick={() => setSelectedGuess(guess)}
                  aria-label={`${guess.word}, 유사도 ${guess.similarity}`}
                >
                  <span className={styles.starCore} />
                  <span className={styles.starLabel}>{guess.word}</span>
                </button>
              ))}

              {selectedGuess && (
                <article className={styles.starInformation}>
                  <div className={styles.informationHeader}>
                    <span
                      className={`${styles.informationStar} ${
                        styles[`star${selectedGuess.spectralType}`]
                      }`}
                    />
                    <div>
                      <span>선택된 단어</span>
                      <strong>{selectedGuess.word}</strong>
                    </div>
                  </div>

                  <dl>
                    <div>
                      <dt>유사도</dt>
                      <dd>{selectedGuess.similarity.toFixed(1)}%</dd>
                    </div>
                    <div>
                      <dt>순위</dt>
                      <dd>{selectedGuess.rank.toLocaleString()}위</dd>
                    </div>
                    <div>
                      <dt>분광형</dt>
                      <dd>{selectedGuess.spectralType}형</dd>
                    </div>
                  </dl>
                </article>
              )}

              <div className={styles.spaceHint}>
                <span>드래그하여 회전</span>
                <span>·</span>
                <span>별을 눌러 정보 확인</span>
              </div>
            </div>
          </section>

          <aside className={styles.sidePanel}>
            <div className={styles.sideHeading}>
              <span className={styles.eyebrow}>TODAY&apos;S SEARCH</span>
              <h2>오늘의 추측</h2>
              <p>
                정답과 의미가 가까운 단어일수록 별이 더 밝고 푸르게
                빛납니다.
              </p>
            </div>

            <form className={styles.searchForm} onSubmit={handleSubmit}>
              <label htmlFor="word-input">단어 입력</label>

              <div className={styles.inputRow}>
                <input
                  id="word-input"
                  type="text"
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  placeholder="추측할 단어를 입력하세요"
                  autoComplete="off"
                />

                <button type="submit">
                  탐색
                  <span aria-hidden="true">↗</span>
                </button>
              </div>
            </form>

            <section className={styles.bestGuessCard}>
              <span>가장 가까운 추측</span>

              <div className={styles.bestGuessContent}>
                <div>
                  <strong>{bestGuess?.word ?? "-"}</strong>
                  <p>{bestGuess?.rank.toLocaleString() ?? "-"}위</p>
                </div>

                <span className={styles.bestSimilarity}>
                  {bestGuess?.similarity.toFixed(1) ?? "0.0"}
                  <small>%</small>
                </span>
              </div>
            </section>

            <section className={styles.historySection}>
              <div className={styles.historyHeader}>
                <h3>최근 추측</h3>
                <span>{guesses.length}개</span>
              </div>

              <div className={styles.historyColumnLabels}>
                <span>단어</span>
                <span>유사도</span>
                <span>순위</span>
              </div>

              <div className={styles.historyList}>
                {[...guesses]
                  .reverse()
                  .slice(0, 8)
                  .map((guess) => (
                    <button
                      type="button"
                      key={guess.id}
                      className={[
                        styles.historyItem,
                        selectedGuess?.id === guess.id
                          ? styles.selectedHistoryItem
                          : "",
                      ].join(" ")}
                      onClick={() => setSelectedGuess(guess)}
                    >
                      <span className={styles.wordCell}>
                        <span
                          className={`${styles.historyStar} ${
                            styles[`star${guess.spectralType}`]
                          }`}
                        />
                        <strong>{guess.word}</strong>
                      </span>

                      <span>{guess.similarity.toFixed(1)}</span>
                      <span>{guess.rank.toLocaleString()}위</span>
                    </button>
                  ))}
              </div>
            </section>
          </aside>
        </div>

        <footer className={styles.legend}>
          <div className={styles.legendTitle}>
            <strong>별의 온도</strong>
            <span>정답과의 의미적 거리</span>
          </div>

          <div className={styles.spectralScale}>
            {spectralTypes.map((type) => (
              <div className={styles.spectralItem} key={type}>
                <span>{type}</span>
                <i
                  className={`${styles.legendStar} ${styles[`star${type}`]}`}
                />
              </div>
            ))}
          </div>

          <div className={styles.direction}>
            <span>정답과 멀리</span>
            <div className={styles.directionLine}>
              <span />
            </div>
            <strong>정답에 가까움</strong>
            <span aria-hidden="true">→</span>
          </div>
        </footer>
      </section>
    </main>
  );
}