"use client";

import {
  useEffect,
  useMemo,
  useState,
} from "react";

import type {
  FormEvent,
} from "react";

import styles from "./page.module.css";

import EmbeddingSpace from "@/features/game/EmbeddingSpace";

import type {
  DisplayGuess,
  SpectralType,
} from "@/features/game/EmbeddingSpace";

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

const spectralTypes: SpectralType[] = [
  "M",
  "K",
  "G",
  "F",
  "A",
  "B",
  "O",
];

/*
 * rank → 분광형
 *
 * 1~10      O
 * 11~50     B
 * 51~150    A
 * 151~350   F
 * 351~650   G
 * 651~1000  K
 * 1001+     M
 */
function getSpectralTypeByRank(
  rank: number | null,
): SpectralType {
  if (
    rank === null ||
    rank > 1000
  ) {
    return "M";
  }

  if (rank <= 10) {
    return "O";
  }

  if (rank <= 50) {
    return "B";
  }

  if (rank <= 150) {
    return "A";
  }

  if (rank <= 350) {
    return "F";
  }

  if (rank <= 650) {
    return "G";
  }

  return "K";
}

/*
 * 백엔드 API Guess를
 * UI용 Guess로 변환.
 *
 * 좌표는 EmbeddingSpace에서
 * rank를 이용해 계산하므로
 * 여기서는 색과 표시용 유사도만 만든다.
 */
function toDisplayGuess(
  guess: ApiGuess,
): DisplayGuess {
  return {
    ...guess,

    similarityPercent:
      guess.similarity * 100,

    spectralType:
      getSpectralTypeByRank(
        guess.rank,
      ),
  };
}

function convertGuesses(
  guesses: ApiGuess[],
): DisplayGuess[] {
  return guesses.map(
    toDisplayGuess,
  );
}

function formatRank(
  rank: number | null,
): string {
  if (rank === null) {
    return "준비 중";
  }

  return `${rank.toLocaleString()}위`;
}

export default function Home() {
  const [
    game,
    setGame,
  ] =
    useState<GameStateResponse | null>(
      null,
    );

  const [
    input,
    setInput,
  ] =
    useState("");

  const [
    guesses,
    setGuesses,
  ] =
    useState<DisplayGuess[]>([]);

  const [
    selectedGuess,
    setSelectedGuess,
  ] =
    useState<DisplayGuess | null>(
      null,
    );

  const [
    isLoading,
    setIsLoading,
  ] =
    useState(false);

  const [
    errorMessage,
    setErrorMessage,
  ] =
    useState("");

  const isFinished =
    game?.status === "won" ||
    game?.status === "abandoned";

  /*
   * 가장 가까운 추측.
   *
   * rank가 있으면 rank 우선.
   * rank가 아직 null이면 similarity fallback.
   */
  const bestGuess =
    useMemo(() => {
      if (
        guesses.length === 0
      ) {
        return null;
      }

      return [
        ...guesses,
      ].sort(
        (
          first,
          second,
        ) => {
          if (
            first.rank !==
              null &&
            second.rank !==
              null
          ) {
            return (
              first.rank -
              second.rank
            );
          }

          if (
            first.rank !==
            null
          ) {
            return -1;
          }

          if (
            second.rank !==
            null
          ) {
            return 1;
          }

          return (
            second.similarity -
            first.similarity
          );
        },
      )[0];
    }, [guesses]);

  /*
   * 새로고침 시 기존 게임 복원.
   */
  useEffect(() => {
    const savedGameId =
      localStorage.getItem(
        "wordOrbitGameId",
      );

    if (!savedGameId) {
      return;
    }

    async function restoreGame(
      gameId: string,
    ) {
      try {
        const restoredGame =
          await getGameState(
            gameId,
          );

        const restoredGuesses =
          convertGuesses(
            restoredGame.guesses,
          );

        setGame(
          restoredGame,
        );

        setGuesses(
          restoredGuesses,
        );

        setSelectedGuess(
          restoredGuesses.length >
            0
            ? restoredGuesses[
                restoredGuesses.length -
                  1
              ]
            : null,
        );
      } catch (error) {
        if (
          error instanceof
            GameApiError &&
          error.code ===
            "GAME_NOT_FOUND"
        ) {
          localStorage.removeItem(
            "wordOrbitGameId",
          );

          return;
        }

        setErrorMessage(
          "기존 게임을 불러오지 못했습니다.",
        );
      }
    }

    void restoreGame(
      savedGameId,
    );
  }, []);

  /*
   * 새 게임 생성.
   */
  async function handleStartGame() {
    if (isLoading) {
      return;
    }

    try {
      setIsLoading(true);

      setErrorMessage("");

      const createdGame =
        await createGame();

      const newGame: GameStateResponse =
        {
          gameId:
            createdGame.gameId,

          status:
            createdGame.status,

          createdAt:
            createdGame.createdAt,

          guessCount: 0,

          guesses: [],

          answer: null,
        };

      setGame(newGame);

      setGuesses([]);

      setSelectedGuess(
        null,
      );

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

  /*
   * 추측 제출.
   */
  async function handleSubmit(
    event: FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault();

    const trimmedWord =
      input.trim();

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

      const submittedGuess =
        await submitGuess(
          game.gameId,
          trimmedWord,
        );

      /*
       * 추측 후 상태를 다시 조회한다.
       *
       * rank가 백엔드에 구현되면
       * 이 응답에 실제 rank가 들어온다.
       */
      const updatedGame =
        await getGameState(
          game.gameId,
        );

      const updatedDisplayGuesses =
        convertGuesses(
          updatedGame.guesses,
        );

      const submittedDisplayGuess =
        updatedDisplayGuesses.find(
          (guess) =>
            guess.guessId ===
            submittedGuess.guessId,
        ) ?? null;

      setGame(
        updatedGame,
      );

      setGuesses(
        updatedDisplayGuesses,
      );

      if (
        submittedDisplayGuess
      ) {
        setSelectedGuess(
          submittedDisplayGuess,
        );
      }

      setInput("");
    } catch (error) {
      await handleApiError(
        error,
      );
    } finally {
      setIsLoading(false);
    }
  }

  async function handleApiError(
    error: unknown,
  ) {
    if (
      !(
        error instanceof
        GameApiError
      )
    ) {
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
        setErrorMessage(
          error.message,
        );
        break;

      case "GAME_NOT_FOUND":
        setErrorMessage(
          "게임을 찾을 수 없습니다. 새 게임을 시작해주세요.",
        );

        setGame(null);

        setGuesses([]);

        setSelectedGuess(
          null,
        );

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
              await getGameState(
                game.gameId,
              );

            const finishedGuesses =
              convertGuesses(
                finishedGame.guesses,
              );

            setGame(
              finishedGame,
            );

            setGuesses(
              finishedGuesses,
            );
          } catch {
            // 기존 오류 메시지 유지
          }
        }

        break;

      default:
        setErrorMessage(
          error.message,
        );
    }
  }

  return (
    <main
      className={styles.page}
    >
      <div
        className={
          styles.backgroundStars
        }
      />

      <section
        className={styles.app}
      >
        <header
          className={
            styles.header
          }
        >
          <div>
            <div
              className={
                styles.logoRow
              }
            >
              <span
                className={
                  styles.logoStar
                }
              >
                ✦
              </span>

              <h1>
                WORD ORBIT
              </h1>
            </div>

            <p>
              Find the hidden word
              in semantic space
            </p>
          </div>

          <nav
            className={
              styles.navigation
            }
            aria-label="주요 메뉴"
          >
            <button
              type="button"
            >
              도움말
            </button>

            <button
              type="button"
            >
              프로젝트
            </button>

            <button
              type="button"
              onClick={
                handleStartGame
              }
              disabled={
                isLoading
              }
            >
              {isLoading
                ? "준비 중..."
                : game
                  ? "새 게임"
                  : "게임 시작"}
            </button>
          </nav>
        </header>

        <div
          className={
            styles.mainContent
          }
        >
          <section
            className={
              styles.spacePanel
            }
          >
            <div
              className={
                styles.spaceHeader
              }
            >
              <div>
                <span
                  className={
                    styles.eyebrow
                  }
                >
                  SEMANTIC
                  UNIVERSE
                </span>

                <h2>
                  단어 임베딩 우주
                </h2>
              </div>

              <div
                className={
                  styles.spaceStatus
                }
              >
                <span
                  className={
                    styles.statusDot
                  }
                />

                {!game
                  ? "게임 시작 전"
                  : isFinished
                    ? "탐색 완료"
                    : "탐색 중"}
              </div>
            </div>

            {/*
             * 기존 2D 우주 영역 대신
             * 3D EmbeddingSpace를 사용한다.
             */}
            <div
              className={
                styles.universe3D
              }
            >
              <EmbeddingSpace
                guesses={
                  guesses
                }
                selectedGuess={
                  selectedGuess
                }
                bestGuess={
                  bestGuess
                }
                onSelectGuess={
                  setSelectedGuess
                }
              />

              {/*
               * 선택 단어 정보는
               * 3D Canvas 위에 오버레이.
               */}
              {selectedGuess && (
                <article
                  className={
                    styles.starInformation3D
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
                      <span>
                        선택된 단어
                      </span>

                      <strong>
                        {
                          selectedGuess.word
                        }
                      </strong>
                    </div>
                  </div>

                  <dl>
                    <div>
                      <dt>
                        유사도
                      </dt>

                      <dd>
                        {selectedGuess.similarityPercent.toFixed(
                          1,
                        )}
                        %
                      </dd>
                    </div>

                    <div>
                      <dt>
                        순위
                      </dt>

                      <dd>
                        {formatRank(
                          selectedGuess.rank,
                        )}
                      </dd>
                    </div>

                    <div>
                      <dt>
                        분광형
                      </dt>

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
                className={
                  styles.spaceHint3D
                }
              >
                <span>
                  드래그하여 회전
                </span>

                <span>·</span>

                <span>
                  휠로 확대/축소
                </span>

                <span>·</span>

                <span>
                  별을 눌러 정보 확인
                </span>
              </div>
            </div>
          </section>

          <aside
            className={
              styles.sidePanel
            }
          >
            <div
              className={
                styles.sideHeading
              }
            >
              <span
                className={
                  styles.eyebrow
                }
              >
                TODAY&apos;S
                SEARCH
              </span>

              <h2>
                오늘의 추측
              </h2>

              <p>
                정답 순위가 높을수록
                별은 더 밝고 푸르게
                빛나며 정답 별 가까이에
                위치합니다.
              </p>
            </div>

            <form
              className={
                styles.searchForm
              }
              onSubmit={
                handleSubmit
              }
            >
              <label
                htmlFor="word-input"
              >
                단어 입력
              </label>

              <div
                className={
                  styles.inputRow
                }
              >
                <input
                  id="word-input"
                  type="text"
                  value={input}
                  onChange={(
                    event,
                  ) =>
                    setInput(
                      event
                        .target
                        .value,
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
                    input.trim()
                        .length ===
                      0
                  }
                >
                  {isLoading
                    ? "탐색 중"
                    : "탐색"}

                  <span
                    aria-hidden="true"
                  >
                    ↗
                  </span>
                </button>
              </div>
            </form>

            {errorMessage && (
              <p
                role="alert"
                className={
                  styles.apiError
                }
              >
                {
                  errorMessage
                }
              </p>
            )}

            {isFinished &&
              game?.answer && (
                <p
                  className={
                    styles.answerReveal
                  }
                >
                  정답은{" "}
                  <strong>
                    {
                      game.answer
                    }
                  </strong>
                  입니다.
                </p>
              )}

            <section
              className={
                styles.bestGuessCard
              }
            >
              <span>
                가장 가까운 추측
              </span>

              <div
                className={
                  styles.bestGuessContent
                }
              >
                <div>
                  <strong>
                    {bestGuess?.word ??
                      "-"}
                  </strong>

                  <p>
                    {bestGuess
                      ? formatRank(
                          bestGuess.rank,
                        )
                      : "-"}
                  </p>
                </div>

                <span
                  className={
                    styles.bestSimilarity
                  }
                >
                  {bestGuess?.similarityPercent.toFixed(
                    1,
                  ) ??
                    "0.0"}

                  <small>
                    %
                  </small>
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
                <h3>
                  최근 추측
                </h3>

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
                <span>
                  단어
                </span>

                <span>
                  유사도
                </span>

                <span>
                  순위
                </span>
              </div>

              <div
                className={
                  styles.historyList
                }
              >
                {[...guesses]
                  .reverse()
                  .slice(0, 8)
                  .map(
                    (
                      guess,
                    ) => (
                      <button
                        type="button"
                        key={
                          guess.guessId
                        }
                        className={[
                          styles.historyItem,

                          selectedGuess?.guessId ===
                          guess.guessId
                            ? styles.selectedHistoryItem
                            : "",
                        ].join(
                          " ",
                        )}
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
                            {
                              guess.word
                            }
                          </strong>
                        </span>

                        <span>
                          {guess.similarityPercent.toFixed(
                            1,
                          )}
                        </span>

                        <span>
                          {guess.rank !==
                          null
                            ? `${guess.rank.toLocaleString()}위`
                            : "-"}
                        </span>
                      </button>
                    ),
                  )}
              </div>
            </section>
          </aside>
        </div>

        <footer
          className={
            styles.legend
          }
        >
          <div
            className={
              styles.legendTitle
            }
          >
            <strong>
              별의 온도
            </strong>

            <span>
              정답과의 순위
            </span>
          </div>

          <div
            className={
              styles.spectralScale
            }
          >
            {spectralTypes.map(
              (type) => (
                <div
                  className={
                    styles.spectralItem
                  }
                  key={type}
                >
                  <span>
                    {type}
                  </span>

                  <i
                    className={`${styles.legendStar} ${
                      styles[
                        `star${type}`
                      ]
                    }`}
                  />
                </div>
              ),
            )}
          </div>

          <div
            className={
              styles.direction
            }
          >
            <span>
              1001위 밖
            </span>

            <div
              className={
                styles.directionLine
              }
            >
              <span />
            </div>

            <strong>
              1~10위
            </strong>

            <span
              aria-hidden="true"
            >
              →
            </span>
          </div>
        </footer>
      </section>
    </main>
  );
}