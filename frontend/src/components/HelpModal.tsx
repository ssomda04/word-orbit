"use client";

import { useRouter } from "next/navigation";

import styles from "./HelpModal.module.css";

type HelpModalProps = {
  onClose: () => void;
};

export default function HelpModal({
  onClose,
}: HelpModalProps) {
  const router = useRouter();

  function handleProjectClick() {
    onClose();
    router.push("/project");
  }

  return (
    <div
      className={styles.overlay}
      onClick={onClose}
      role="presentation"
    >
      <section
        className={styles.modal}
        role="dialog"
        aria-modal="true"
        aria-labelledby="help-modal-title"
        onClick={(event) =>
          event.stopPropagation()
        }
      >
        <header
          className={styles.header}
        >
          <div>
            <span
              className={styles.eyebrow}
            >
              HOW TO PLAY
            </span>

            <h2 id="help-modal-title">
              Word Orbit 탐험 가이드
            </h2>

            <p>
              단어의 의미를 따라 우주를
              탐험하고 숨겨진 정답을
              찾아보세요.
            </p>
          </div>

          <button
            type="button"
            className={styles.close}
            onClick={onClose}
            aria-label="도움말 닫기"
          >
            ×
          </button>
        </header>

        <div
          className={styles.content}
        >
          <section
            className={styles.steps}
          >
            <article
              className={styles.step}
            >
              <span
                className={styles.stepNumber}
              >
                01
              </span>

              <div>
                <h3>게임 시작</h3>

                <p>
                  화면 오른쪽 위의
                  <strong> 게임 시작 </strong>
                  버튼을 눌러 새로운 단어
                  탐색을 시작합니다.
                </p>
              </div>
            </article>

            <article
              className={styles.step}
            >
              <span
                className={styles.stepNumber}
              >
                02
              </span>

              <div>
                <h3>단어 추측</h3>

                <p>
                  오른쪽의 단어 입력창에
                  정답일 것 같은 단어를
                  입력하고
                  <strong> 탐색 </strong>
                  버튼을 눌러보세요.
                </p>
              </div>
            </article>

            <article
              className={styles.step}
            >
              <span
                className={styles.stepNumber}
              >
                03
              </span>

              <div>
                <h3>
                  유사도와 순위 확인
                </h3>

                <p>
                  입력한 단어와 정답 사이의
                  의미적 유사도와 순위가
                  표시됩니다. 순위 숫자가
                  <strong> 1위에 가까울수록 </strong>
                  정답과 의미적으로 더 가까운
                  단어입니다.
                </p>
              </div>
            </article>

            <article
              className={styles.step}
            >
              <span
                className={styles.stepNumber}
              >
                04
              </span>

              <div>
                <h3>
                  별을 따라 정답 찾기
                </h3>

                <p>
                  추측한 단어는 임베딩 우주에
                  별로 나타납니다. 정답에
                  가까운 단어일수록 더 밝고
                  푸르게 표현되며, 정답 별에
                  가까운 곳에 위치합니다.
                </p>
              </div>
            </article>
          </section>

          <section
            className={styles.spaceGuide}
          >
            <div
              className={styles.guideItem}
            >
              <span
                className={styles.guideIcon}
              >
                ↔
              </span>

              <div>
                <strong>
                  드래그
                </strong>

                <span>
                  우주를 회전합니다.
                </span>
              </div>
            </div>

            <div
              className={styles.guideItem}
            >
              <span
                className={styles.guideIcon}
              >
                ⊕
              </span>

              <div>
                <strong>
                  마우스 휠
                </strong>

                <span>
                  화면을 확대하거나
                  축소합니다.
                </span>
              </div>
            </div>

            <div
              className={styles.guideItem}
            >
              <span
                className={styles.guideIcon}
              >
                ✦
              </span>

              <div>
                <strong>
                  별 선택
                </strong>

                <span>
                  해당 단어의 유사도와
                  순위를 확인합니다.
                </span>
              </div>
            </div>
          </section>

          <section
            className={styles.goal}>
            <span>MISSION</span>

            <div>
              <strong>
                정답 단어를 찾아
                1위에 도달하세요.
              </strong>

              <p>
                가까운 단어에서 의미적
                단서를 발견하고 다음
                추측을 이어가면 됩니다.
              </p>
            </div>
          </section>

          <section
            className={styles.projectGuide}
          >
            <div>
              <span
                className={styles.projectEyebrow}
              >
                BEHIND THE ORBIT
              </span>

              <h3>
                이 게임의 원리가
                궁금한가요?
              </h3>

              <p>
                Word Embedding과 코사인
                유사도, 단어들이 우주처럼
                배치되는 원리가 궁금하다면
                상단의
                <strong> 프로젝트 </strong>
                메뉴를 확인해 보세요.
              </p>
            </div>

            <button
              type="button"
              className={styles.projectButton}
              onClick={
                handleProjectClick
              }
            >
              프로젝트 알아보기
              <span
                aria-hidden="true"
              >
                →
              </span>
            </button>
          </section>
        </div>
      </section>
    </div>
  );
}