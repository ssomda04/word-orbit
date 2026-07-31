"use client";

import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";

import styles from "./page.module.css";

import {
  createGame,
  GameApiError,
  getGameState,
  submitGuess,
} from "@/lib/gameApi";

import type {
  GameStateResponse,
  Guess as ApiGuess,
} from "@/types/api";

type SpectralType = "M" | "K" | "G" | "F" | "A" | "B" | "O";

/**
 * 백엔드 Guess에 화면 표시용 정보를 추가한 타입이다.
 *
 * spectralType: 유사도에 따른 별의 종류
 * x, y: coordinate가 구현되기 전까지 사용할 임시 화면 좌표
 */
type DisplayGuess = ApiGuess & {
  spectralType: SpectralType;
  x: number;
  y: number;
};

const spectralTypes: SpectralType[] = [
  "M",
  "K",
  "G",
  "F",
  "A",
  "B",
  "O",
];

/**
 * 유사도 백분율을 별의 분광형으로 변환한다.
 */
function getSpectralType(
  similarityPercent: number,
): SpectralType {
  if (similarityPercent >= 95) return "O";
  if (similarityPercent >= 88) return "B";
  if (similarityPercent >= 80) return "A";
  if (similarityPercent >= 70) return "F";
  if (similarityPercent >= 55) return "G";
  if (similarityPercent >= 35) return "K";

  return "M";
}

/**
 * API Guess를 기존 우주 UI에서 사용할 형태로 변환한다.
 *
 * 백엔드 similarity 범위는 -1 ~ 1이므로
 * 화면 표시용으로 100을 곱한다.
 *
 * coordinate는 아직 null이므로 index 기반 임시 좌표를 사용한다.
 */
function toDisplayGuess(
  guess: ApiGuess,
  index: number,
): DisplayGuess {
  const similarityPercent = guess.similarity * 100;

  return {
    ...guess,
    similarity: similarityPercent,
    spectralType: getSpectralType(similarityPercent),

    // coordinate 구현 전 임시 시각화 좌표
    x: 14 + ((index * 19 + 11) % 72),
    y: 16 + ((index * 23 + 7) % 66),
  };
}

function convertGuesses(
  guesses: ApiGuess[],
): DisplayGuess[] {
  return guesses.map((guess, index) =>
    toDisplayGuess(guess, index),
  );
}

export default function Home() {
  const [game, setGame] =
    useState<GameStateResponse | null>(null);

  const [input, setInput] = useState("");

  const [guesses, setGuesses] =
    useState<DisplayGuess[]>([]);

  const [selectedGuess, setSelectedGuess] =
    useState<DisplayGuess | null>(null);

  const [isLoading, setIsLoading] = useState(false);

  const [errorMessage, setErrorMessage] =
    useState("");

  const isFinished =
    game?.status === "won" ||
    game?.status === "abandoned";

  /**
   * 현재 가장 유사도가 높은 추측이다.
   */
  const bestGuess = useMemo(() => {
    if (guesses.length === 0) {
      return null;
    }

    return [...guesses].sort(
      (first, second) =>
        second.similarity - first.similarity,
    )[0];
  }, [guesses]);

  /**
   * 새로고침했을 때 이전 gameId가 있으면
   * 백엔드에서 게임 상태를 다시 불러온다.
   */
  useEffect(() => {
  const savedGameId = localStorage.getItem("wordOrbitGameId");

  if (!savedGameId) {
    return;
  }

  async function restoreGame(gameId: string) {
    try {
      const restoredGame = await getGameState(gameId);

      const restoredGuesses = convertGuesses(
        restoredGame.guesses,
      );

      setGame(restoredGame);
      setGuesses(restoredGuesses);
      setSelectedGuess(restoredGuesses.at(-1) ?? null);
    } catch (error) {
      if (
        error instanceof GameApiError &&
        error.code === "GAME_NOT_FOUND"
      ) {
        localStorage.removeItem("wordOrbitGameId");
        return;
      }

      setErrorMessage("기존 게임을 불러오지 못했습니다.");
    }
  }

  void restoreGame(savedGameId);
}, []);

  /**
   * 새 게임 생성
   */
  async function handleStartGame() {
    if (isLoading) {
      return;
    }

    try {
      setIsLoading(true);
      setErrorMessage("");

      const createdGame = await createGame();

      const newGame: GameStateResponse = {
        gameId: createdGame.gameId,
        status: createdGame.status,
        createdAt: createdGame.createdAt,
        guessCount: 0,
        guesses: [],
        answer: null,
      };

      setGame(newGame);
      setGuesses([]);
      setSelectedGuess(null);
      setInput("");

      localStorage.setItem(
        "wordOrbitGameId",
        createdGame.gameId,
      );
    } catch (error) {
      setErrorMessage(
        error instanceof Error
          ? error.message
          : "게임 생성에 실패했습니다.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  /**
   * 단어 추측 제출
   */
  async function handleSubmit(
    event: FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault();

    const trimmedWord = input.trim();

    if (
      !trimmedWord ||
      !game ||
      isLoading ||
      isFinished
    ) {
      return;
    }

    try {
      setIsLoading(true);
      setErrorMessage("");

      const submittedGuess = await submitGuess(
        game.gameId,
        trimmedWord,
      );

      /**
       * 추측 제출 후 상태를 다시 조회한다.
       *
       * 이렇게 하면 중복 추측, guessCount,
       * 게임 종료 상태가 백엔드와 정확히 일치한다.
       */
      const updatedGame = await getGameState(
        game.gameId,
      );

      const updatedDisplayGuesses =
        convertGuesses(updatedGame.guesses);

      const submittedDisplayGuess =
        updatedDisplayGuesses.find(
          (guess) =>
            guess.guessId ===
            submittedGuess.guessId,
        ) ?? null;

      setGame(updatedGame);
      setGuesses(updatedDisplayGuesses);

      if (submittedDisplayGuess) {
        setSelectedGuess(
          submittedDisplayGuess,
        );
      }

      setInput("");
    } catch (error) {
      await handleApiError(error);
    } finally {
      setIsLoading(false);
    }
  }

  /**
   * API 오류 처리
   */
  async function handleApiError(error: unknown) {
    if (!(error instanceof GameApiError)) {
      setErrorMessage(
        "서버와 통신하는 중 오류가 발생했습니다.",
      );

      return;
    }

    switch (error.code) {
      case "INVALID_INPUT":
        setErrorMessage(
          "추측할 단어를 입력해주세요.",
        );
        break;

      case "INVALID_WORD":
        setErrorMessage(error.message);
        break;

      case "GAME_NOT_FOUND":
        setErrorMessage(
          "게임을 찾을 수 없습니다. 새 게임을 시작해주세요.",
        );

        setGame(null);
        setGuesses([]);
        setSelectedGuess(null);

        localStorage.removeItem(
          "wordOrbitGameId",
        );
        break;

      case "GAME_ALREADY_FINISHED":
        setErrorMessage(
          "이미 종료된 게임입니다.",
        );

        if (game) {
          try {
            const finishedGame =
              await getGameState(game.gameId);

            const finishedGuesses =
              convertGuesses(
                finishedGame.guesses,
              );

            setGame(finishedGame);
            setGuesses(finishedGuesses);
          } catch {
            // 상태 재조회에 실패해도 기존 오류 메시지는 유지한다.
          }
        }
        break;

      default:
        setErrorMessage(error.message);
    }
  }

  return (
    <main className={styles.page}>
      <div className={styles.backgroundStars} />

      <section className={styles.app}>
        <header className={styles.header}>
          <div>
            <div className={styles.logoRow}>
              <span className={styles.logoStar}>
                ✦
              </span>

              <h1>WORD ORBIT</h1>
            </div>

            <p>
              Find the hidden word in semantic
              space
            </p>
          </div>

          <nav
            className={styles.navigation}
            aria-label="주요 메뉴"
          >
            <button type="button">
              도움말
            </button>

            <button type="button">
              프로젝트
            </button>

            <button
              type="button"
              onClick={handleStartGame}
              disabled={isLoading}
            >
              {isLoading
                ? "준비 중..."
                : game
                  ? "새 게임"
                  : "게임 시작"}
            </button>
          </nav>
        </header>

        <div className={styles.mainContent}>
          <section
            className={styles.spacePanel}
          >
            <div
              className={styles.spaceHeader}
            >
              <div>
                <span
                  className={styles.eyebrow}
                >
                  SEMANTIC UNIVERSE
                </span>

                <h2>단어 임베딩 우주</h2>
              </div>

              <div
                className={styles.spaceStatus}
              >
                <span
                  className={styles.statusDot}
                />

                {!game
                  ? "게임 시작 전"
                  : isFinished
                    ? "탐색 완료"
                    : "탐색 중"}
              </div>
            </div>

            <div className={styles.universe}>
              <div
                className={styles.orbitLarge}
              />

              <div
                className={styles.orbitSmall}
              />

              <svg
                className={
                  styles.constellationLines
                }
                viewBox="0 0 100 100"
                preserveAspectRatio="none"
                aria-hidden="true"
              >
                {guesses
                  .slice(1)
                  .map((guess, index) => {
                    const previousGuess =
                      guesses[index];

                    return (
                      <line
                        key={`${previousGuess.guessId}-${guess.guessId}`}
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
                <span
                  className={styles.answerCore}
                />

                <span
                  className={
                    styles.answerPulse
                  }
                />
              </button>

              {guesses.map((guess) => (
                <button
                  type="button"
                  key={guess.guessId}
                  className={[
                    styles.wordStar,
                    styles[
                      `star${guess.spectralType}`
                    ],
                    selectedGuess?.guessId ===
                    guess.guessId
                      ? styles.selectedStar
                      : "",
                    bestGuess?.guessId ===
                    guess.guessId
                      ? styles.bestStar
                      : "",
                  ].join(" ")}
                  style={{
                    left: `${guess.x}%`,
                    top: `${guess.y}%`,
                  }}
                  onClick={() =>
                    setSelectedGuess(guess)
                  }
                  aria-label={`${guess.word}, 유사도 ${guess.similarity.toFixed(
                    1,
                  )}`}
                >
                  <span
                    className={
                      styles.starCore
                    }
                  />

                  <span
                    className={
                      styles.starLabel
                    }
                  >
                    {guess.word}
                  </span>
                </button>
              ))}

              {selectedGuess && (
                <article
                  className={
                    styles.starInformation
                  }
                >
                  <div
                    className={
                      styles.informationHeader
                    }
                  >
                    <span
                      className={`${styles.informationStar} ${
                        styles[
                          `star${selectedGuess.spectralType}`
                        ]
                      }`}
                    />

                    <div>
                      <span>선택된 단어</span>
                      <strong>
                        {selectedGuess.word}
                      </strong>
                    </div>
                  </div>

                  <dl>
                    <div>
                      <dt>유사도</dt>

                      <dd>
                        {selectedGuess.similarity.toFixed(
                          1,
                        )}
                        %
                      </dd>
                    </div>

                    <div>
                      <dt>순위</dt>

                      <dd>
                        {selectedGuess.rank !==
                        null
                          ? `${selectedGuess.rank.toLocaleString()}위`
                          : "준비 중"}
                      </dd>
                    </div>

                    <div>
                      <dt>분광형</dt>

                      <dd>
                        {
                          selectedGuess.spectralType
                        }
                        형
                      </dd>
                    </div>
                  </dl>
                </article>
              )}

              <div
                className={styles.spaceHint}
              >
                <span>별을 눌러 정보 확인</span>
                <span>·</span>
                <span>
                  좌표는 현재 임시 배치
                </span>
              </div>
            </div>
          </section>

          <aside
            className={styles.sidePanel}
          >
            <div
              className={styles.sideHeading}
            >
              <span
                className={styles.eyebrow}
              >
                TODAY&apos;S SEARCH
              </span>

              <h2>오늘의 추측</h2>

              <p>
                정답과 의미가 가까운
                단어일수록 별이 더 밝고
                푸르게 빛납니다.
              </p>
            </div>

            <form
              className={styles.searchForm}
              onSubmit={handleSubmit}
            >
              <label htmlFor="word-input">
                단어 입력
              </label>

              <div
                className={styles.inputRow}
              >
                <input
                  id="word-input"
                  type="text"
                  value={input}
                  onChange={(event) =>
                    setInput(
                      event.target.value,
                    )
                  }
                  placeholder={
                    !game
                      ? "먼저 게임을 시작하세요"
                      : isFinished
                        ? "게임이 종료되었습니다"
                        : "추측할 단어를 입력하세요"
                  }
                  autoComplete="off"
                  disabled={
                    !game ||
                    isLoading ||
                    isFinished
                  }
                />

                <button
                  type="submit"
                  disabled={
                    !game ||
                    isLoading ||
                    isFinished ||
                    input.trim().length === 0
                  }
                >
                  {isLoading
                    ? "탐색 중"
                    : "탐색"}

                  <span aria-hidden="true">
                    ↗
                  </span>
                </button>
              </div>
            </form>

            {errorMessage && (
              <p
                role="alert"
                style={{
                  marginTop: "12px",
                }}
              >
                {errorMessage}
              </p>
            )}

            {isFinished && game?.answer && (
              <p
                style={{
                  marginTop: "12px",
                }}
              >
                정답은{" "}
                <strong>
                  {game.answer}
                </strong>
                입니다.
              </p>
            )}

            <section
              className={
                styles.bestGuessCard
              }
            >
              <span>가장 가까운 추측</span>

              <div
                className={
                  styles.bestGuessContent
                }
              >
                <div>
                  <strong>
                    {bestGuess?.word ?? "-"}
                  </strong>

                  <p>
                    {bestGuess?.rank !==
                      null &&
                    bestGuess?.rank !==
                      undefined
                      ? `${bestGuess.rank.toLocaleString()}위`
                      : "순위 준비 중"}
                  </p>
                </div>

                <span
                  className={
                    styles.bestSimilarity
                  }
                >
                  {bestGuess?.similarity.toFixed(
                    1,
                  ) ?? "0.0"}

                  <small>%</small>
                </span>
              </div>
            </section>

            <section
              className={
                styles.historySection
              }
            >
              <div
                className={
                  styles.historyHeader
                }
              >
                <h3>최근 추측</h3>

                <span>
                  {game?.guessCount ??
                    guesses.length}
                  개
                </span>
              </div>

              <div
                className={
                  styles.historyColumnLabels
                }
              >
                <span>단어</span>
                <span>유사도</span>
                <span>순위</span>
              </div>

              <div
                className={
                  styles.historyList
                }
              >
                {[...guesses]
                  .reverse()
                  .slice(0, 8)
                  .map((guess) => (
                    <button
                      type="button"
                      key={guess.guessId}
                      className={[
                        styles.historyItem,
                        selectedGuess?.guessId ===
                        guess.guessId
                          ? styles.selectedHistoryItem
                          : "",
                      ].join(" ")}
                      onClick={() =>
                        setSelectedGuess(
                          guess,
                        )
                      }
                    >
                      <span
                        className={
                          styles.wordCell
                        }
                      >
                        <span
                          className={`${styles.historyStar} ${
                            styles[
                              `star${guess.spectralType}`
                            ]
                          }`}
                        />

                        <strong>
                          {guess.word}
                        </strong>
                      </span>

                      <span>
                        {guess.similarity.toFixed(
                          1,
                        )}
                      </span>

                      <span>
                        {guess.rank !== null
                          ? `${guess.rank.toLocaleString()}위`
                          : "-"}
                      </span>
                    </button>
                  ))}
              </div>
            </section>
          </aside>
        </div>

        <footer className={styles.legend}>
          <div
            className={styles.legendTitle}
          >
            <strong>별의 온도</strong>

            <span>
              정답과의 의미적 거리
            </span>
          </div>

          <div
            className={styles.spectralScale}
          >
            {spectralTypes.map((type) => (
              <div
                className={
                  styles.spectralItem
                }
                key={type}
              >
                <span>{type}</span>

                <i
                  className={`${styles.legendStar} ${
                    styles[`star${type}`]
                  }`}
                />
              </div>
            ))}
          </div>

          <div
            className={styles.direction}
          >
            <span>정답과 멀리</span>

            <div
              className={
                styles.directionLine
              }
            >
              <span />
            </div>

            <strong>
              정답에 가까움
            </strong>

            <span aria-hidden="true">
              →
            </span>
          </div>
        </footer>
      </section>
    </main>
  );
}