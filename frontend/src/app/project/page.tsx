import Link from "next/link";

import styles from "./page.module.css";

export default function ProjectPage() {
  return (
    <main className={styles.page}>
      <div className={styles.backgroundGlow} />

      <header className={styles.header}>
        <Link href="/" className={styles.logo}>
          <span>✦</span>
          WORD ORBIT
        </Link>

        <Link href="/" className={styles.backButton}>
          ← 게임으로 돌아가기
        </Link>
      </header>

      <section className={styles.hero}>
        <span className={styles.eyebrow}>
          BEHIND THE ORBIT
        </span>

        <h1>
          단어는 어떻게
          <br />
          <span>우주가 될까요?</span>
        </h1>

        <p>
          Word Orbit은 단어를 단순한 글자가 아니라 의미를 가진 벡터로
          바라봅니다.
          <br />
          단어가 의미 공간에 배치되는 과정부터 두 단어의 유사도를 측정하는
          방법까지, 게임 속에 숨어 있는 수학적 원리를 소개합니다.
        </p>
      </section>

      <section className={styles.content}>
        {/* One-hot → Embedding */}
        <article className={styles.section}>
          <div className={styles.sectionBody}>
            <span className={styles.sectionLabel}>
              FROM ONE-HOT TO EMBEDDING
            </span>

            <h2>단어를 의미가 있는 벡터로 표현하기</h2>

            <p className={styles.description}>
              컴퓨터가 단어를 표현하는 가장 단순한 방법 중 하나는 단어마다
              고유한 위치를 정하고, 해당 위치만 1로 표시하는
              <strong> One-hot Vector</strong>를 사용하는 것입니다.
            </p>

            <div className={styles.exampleBox}>
              <div className={styles.exampleTitle}>
                ONE-HOT REPRESENTATION
              </div>

              <div className={styles.wordExample}>
                <div>
                  <span>사과</span>
                  <code>[1, 0, 0, 0, ...]</code>
                </div>

                <div>
                  <span>바나나</span>
                  <code>[0, 1, 0, 0, ...]</code>
                </div>

                <div>
                  <span>자동차</span>
                  <code>[0, 0, 1, 0, ...]</code>
                </div>
              </div>

              <p>
                이 표현에서는 &quot;사과&quot;와 &quot;바나나&quot;가
                의미적으로 비슷하더라도 벡터 사이에 특별한 관계가 존재하지
                않습니다.
                <strong>
                  {" "}
                  즉, 단어를 구별할 수는 있지만 단어 사이의 의미적 유사성을
                  표현하기는 어렵습니다.
                </strong>
              </p>
            </div>

            <p className={styles.description}>
              Word Embedding은 이러한 한계를 해결하기 위해 각 단어를
              고차원의 One-hot Vector 대신 더 작은 차원의 연속적인 벡터로
              표현합니다. 이때 단어를 임베딩 공간으로 옮기는 변환을
              <strong> Embedding Matrix W</strong>로 나타낼 수 있습니다.
            </p>

            <div className={styles.formulaBox}>
              <span>EMBEDDING TRANSFORMATION</span>

              <strong>vᵢ = xᵢᵀW</strong>

              <p>
                One-hot Vector xᵢ와 Embedding Matrix W를 곱하면, W에서
                i번째 단어에 해당하는 행이 선택되어 그 단어의 Embedding
                Vector vᵢ를 얻습니다.
                <br />
                즉, 단어의 단순한 번호 표현을 의미를 학습할 수 있는 연속적인
                벡터 표현으로 바꾸는 과정입니다.
              </p>
            </div>

            <p className={styles.description}>
              처음의 Embedding Matrix에는 임의의 값이 들어 있기 때문에 단어
              벡터들도 의미 없이 흩어져 있습니다. 이후 학습 과정에서 W가
              반복적으로 수정되면서 단어 벡터 사이에 의미적인 구조가
              형성됩니다.
            </p>

            <p className={styles.description}>
              이렇게 학습된 공간에서는 단어의 관계 자체가 하나의 방향으로
              나타나기도 합니다. 대표적으로 알려진
              <strong> king − man + woman ≈ queen</strong>과 같은 벡터
              연산은 단어의 의미적 관계가 임베딩 공간의 기하학적 구조로
              표현될 수 있음을 보여줍니다.
            </p>
          </div>
        </article>

        {/* Context learning */}
        <article className={styles.section}>
          <div className={styles.sectionBody}>
            <span className={styles.sectionLabel}>
              LEARNING FROM CONTEXT
            </span>

            <h2>문맥으로 단어의 위치를 학습하기</h2>

            <p className={styles.description}>
              그렇다면 임베딩 공간에서 어떤 단어들이 서로 가까워져야 하는지는
              어떻게 결정할까요? Word2Vec의 핵심 아이디어는
              <strong>
                {" "}
                비슷한 문맥에서 자주 등장하는 단어는 의미적으로도 관련될
                가능성이 높다
              </strong>
              는 것입니다.
            </p>

            <div className={styles.sentenceBox}>
              <p>
                “전자는 <strong>낮은 [에너지] 상태</strong>로 이동한다.”
              </p>

              <p>
                “계는 <strong>낮은 [에너지] 상태</strong>를 선택한다.”
              </p>

              <p>
                “입자는 <strong>낮은 [에너지] 상태</strong>를 선호한다.”
              </p>
            </div>

            <p className={styles.description}>
              예를 들어 &quot;낮은 [ ] 상태&quot;라는 문맥을 표현하기 위해
              주변 단어의 벡터를 이용해 하나의
              <strong> Context Vector</strong>를 만들 수 있습니다.
            </p>

            <div className={styles.formulaBox}>
              <span>CONTEXT VECTOR</span>

              <strong>v꜀ = (v낮은 + v상태) / 2</strong>

              <p>
                주변 단어의 벡터를 이용해 문맥 자체도 하나의 벡터로 표현할 수
                있습니다.
              </p>
            </div>

            <p className={styles.description}>
              그리고 특정 단어의 벡터 vᵢ와 Context Vector v꜀의
              <strong> 내적(Dot Product)</strong>을 계산합니다.
              내적이 크다는 것은 두 벡터가 더 잘 정렬되어 있다는 뜻입니다.
            </p>

            <p className={styles.description}>
              따라서 &quot;낮은 [ ] 상태&quot;라는 문맥에서
              &quot;에너지&quot;가 반복해서 관측된다면, 학습 과정에서는
              &quot;에너지&quot; 벡터가 이 Context Vector와 더 잘 정렬되는
              방향으로 이동하게 됩니다.
            </p>

            <div className={styles.theoryChain}>
              <div>
                <span>DOT PRODUCT</span>
                <strong>내적 증가</strong>
                <p>vᵢ · v꜀ ↑</p>
              </div>

              <div>→</div>

              <div>
                <span>ENERGY</span>
                <strong>에너지 감소</strong>
                <p>E(i,c) = −vᵢ · v꜀ ↓</p>
              </div>

              <div>→</div>

              <div>
                <span>PROBABILITY</span>
                <strong>등장 확률 증가</strong>
                <p>p(vᵢ | v꜀) ↑</p>
              </div>
            </div>
          </div>
        </article>

        {/* Softmax / likelihood */}
        <article className={styles.section}>
          <div className={styles.sectionBody}>
            <span className={styles.sectionLabel}>
              SOFTMAX &amp; LEARNING
            </span>

            <h2>내적 점수를 확률로 바꾸기</h2>

            <p className={styles.description}>
              하지만 vᵢ · v꜀는 아직 단순한 점수입니다. 모델은 특정 문맥에서
              여러 후보 단어 중 어떤 단어가 등장할 가능성이 높은지를 표현해야
              하므로, 이 점수를
              <strong> 확률 분포</strong>로 변환할 필요가 있습니다.
            </p>

            <p className={styles.description}>
              이때 사용되는 함수가 <strong>Softmax</strong>입니다.
            </p>

            <div className={styles.formulaBox}>
              <span>SOFTMAX</span>

              <strong>
                p(vᵢ | v꜀) = exp(vᵢ · v꜀) / Σⱼ exp(vⱼ · v꜀)
              </strong>

              <p>
                Softmax를 적용하면 모든 후보 단어의 확률 합은 1이 됩니다.
                Context Vector와 더 잘 정렬된 단어는 더 큰 내적값을 가지므로
                더 높은 확률을 받습니다.
              </p>
            </div>

            <p className={styles.description}>
              모델은 실제 말뭉치에서 관측된 단어 분포와 자신이 예측한 확률
              분포가 가까워지도록 Embedding Matrix를 수정합니다. 즉,
              <strong> Likelihood</strong>가 높아지는 방향으로 벡터 공간을
              조정합니다.
            </p>

            <p className={styles.description}>
              실제 데이터에서 어떤 단어가 모델의 예상보다 자주 등장했다면 그
              단어 벡터는 Context Vector 방향으로 이동하고, 반대로 모델이
              지나치게 높은 확률을 부여했다면 해당 방향에서 멀어지도록
              조정됩니다.
            </p>

            <div className={styles.researchNote}>
              Word2Vec의 학습은 단순히 단어를 숫자로 저장하는 과정이 아니라,
              <strong>
                {" "}
                문맥과 관측 확률을 이용해 벡터 공간의 기하학적 구조를
                만들어가는 최적화 과정
              </strong>
              으로 볼 수 있습니다.
            </div>
          </div>
        </article>

        {/* Cosine similarity */}
        <article className={styles.section}>
          <div className={styles.sectionBody}>
            <span className={styles.sectionLabel}>
              COSINE SIMILARITY
            </span>

            <h2>코사인 유사도로 단어 사이의 관계 측정하기</h2>

            <p className={styles.description}>
              학습이 끝난 뒤에는 각 단어가 임베딩 공간의 하나의 벡터로
              표현됩니다. 이제 두 단어 벡터가 얼마나 비슷한지를 측정하면
              <strong> 단어 사이의 의미적 유사성</strong>을 수치로 나타낼 수
              있습니다.
            </p>

            <p className={styles.description}>
              Word Orbit에서는 정답 단어와 사용자가 추측한 단어의 관계를
              비교하기 위해
              <strong> Cosine Similarity</strong>를 사용합니다.
            </p>

            <div className={styles.formulaBox}>
              <span>COSINE SIMILARITY</span>

              <strong>sim(a, b) = (a · b) / (‖a‖ ‖b‖)</strong>

              <p>
                두 벡터의 내적을 각각의 벡터 크기로 정규화하여 두 벡터가
                얼마나 비슷한 방향을 향하는지를 측정합니다.
              </p>
            </div>

            <p className={styles.description}>
              단순한 내적은 두 벡터의 방향뿐 아니라 벡터의 크기에도 영향을
              받습니다. 반면 코사인 유사도는 각 벡터의 크기로 나누어
              정규화하기 때문에
              <strong>
                {" "}
                벡터의 길이보다 두 벡터의 방향 관계에 집중
              </strong>
              할 수 있습니다.
            </p>

            <div className={styles.cosineScale}>
              <div>
                <strong>−1</strong>
                <span>반대 방향</span>
              </div>

              <div>
                <strong>0</strong>
                <span>직교</span>
              </div>

              <div>
                <strong>1</strong>
                <span>같은 방향</span>
              </div>
            </div>

            <div className={styles.cosineVisualGrid}>
              {/* 높은 유사도 */}
              <div className={styles.vectorExample}>
                <div className={styles.vectorExampleHeader}>
                  <span>HIGH SIMILARITY</span>
                  <strong>방향이 비슷한 두 벡터</strong>
                  <p>
                    두 벡터 사이의 각도가 작을수록 코사인 유사도는 1에
                    가까워집니다.
                  </p>
                </div>

                <div className={styles.vectorCanvas}>
                  <svg
                    viewBox="0 0 420 260"
                    className={styles.vectorSvg}
                    role="img"
                    aria-label="흰색 벡터는 70도, 파란색 벡터는 60도인 높은 코사인 유사도 예시"
                  >
                    <defs>
                      <marker
                        id="arrowWhiteHigh"
                        markerWidth="5"
                        markerHeight="5"
                        refX="4.5"
                        refY="2.5"
                        orient="auto"
                        markerUnits="strokeWidth"
                      >
                        <path d="M0,0 L5,2.5 L0,5 Z" fill="#f3f7ff" />
                      </marker>

                      <marker
                        id="arrowBlueHigh"
                        markerWidth="5"
                        markerHeight="5"
                        refX="4.5"
                        refY="2.5"
                        orient="auto"
                        markerUnits="strokeWidth"
                      >
                        <path d="M0,0 L5,2.5 L0,5 Z" fill="#86b6ff" />
                      </marker>
                    </defs>

                    <line
                      x1="80"
                      y1="205"
                      x2="80"
                      y2="45"
                      className={styles.vectorAxis}
                    />

                    <line
                      x1="80"
                      y1="205"
                      x2="355"
                      y2="205"
                      className={styles.vectorAxis}
                    />

                    <circle
                      cx="80"
                      cy="205"
                      r="4"
                      className={styles.vectorOrigin}
                    />

                    {/* 흰색 기준 벡터: 70° */}
                    <line
                      x1="80"
                      y1="205"
                      x2="135"
                      y2="55"
                      className={styles.vectorLineWhite}
                      style={{ strokeWidth: 2 }}
                      markerEnd="url(#arrowWhiteHigh)"
                    />

                    {/* 파란색 비교 벡터: 60° */}
                    <line
                      x1="80"
                      y1="205"
                      x2="160"
                      y2="66"
                      className={styles.vectorLineBlue}
                      style={{ strokeWidth: 2 }}
                      markerEnd="url(#arrowBlueHigh)"
                    />

                    {/* 두 벡터 사이 약 10° */}
                    <path
                      d="M 110 153 A 60 60 0 0 0 100.5 148.6"
                      className={styles.angleArc}
                    />

                    <text x="120" y="170" className={styles.angleText}>
                      θ
                    </text>

                    <text
                      x="95"
                      y="45"
                      className={styles.vectorLabelWhite}
                    >
                      vector B
                    </text>

                    <text
                      x="168"
                      y="68"
                      className={styles.vectorLabelBlue}
                    >
                      vector A
                    </text>
                  </svg>
                </div>

                <div className={styles.cosineResult}>
                  <span>작은 θ</span>
                  <strong>cos θ → 1</strong>
                  <p>의미적 방향이 유사함</p>
                </div>
              </div>

              {/* 낮은 유사도 */}
              <div className={styles.vectorExample}>
                <div className={styles.vectorExampleHeader}>
                  <span>LOW SIMILARITY</span>
                  <strong>방향 차이가 큰 두 벡터</strong>
                  <p>
                    두 벡터 사이의 각도가 커질수록 코사인 유사도는 낮아집니다.
                  </p>
                </div>

                <div className={styles.vectorCanvas}>
                  <svg
                    viewBox="0 0 420 260"
                    className={styles.vectorSvg}
                    role="img"
                    aria-label="흰색 벡터는 70도, 파란색 벡터는 20도인 낮은 코사인 유사도 예시"
                  >
                    <defs>
                      <marker
                        id="arrowWhiteLow"
                        markerWidth="5"
                        markerHeight="5"
                        refX="4.5"
                        refY="2.5"
                        orient="auto"
                        markerUnits="strokeWidth"
                      >
                        <path d="M0,0 L5,2.5 L0,5 Z" fill="#f3f7ff" />
                      </marker>

                      <marker
                        id="arrowBlueLow"
                        markerWidth="5"
                        markerHeight="5"
                        refX="4.5"
                        refY="2.5"
                        orient="auto"
                        markerUnits="strokeWidth"
                      >
                        <path d="M0,0 L5,2.5 L0,5 Z" fill="#86b6ff" />
                      </marker>
                    </defs>

                    <line
                      x1="80"
                      y1="205"
                      x2="80"
                      y2="45"
                      className={styles.vectorAxis}
                    />

                    <line
                      x1="80"
                      y1="205"
                      x2="355"
                      y2="205"
                      className={styles.vectorAxis}
                    />

                    <circle
                      cx="80"
                      cy="205"
                      r="4"
                      className={styles.vectorOrigin}
                    />

                    {/* 흰색 기준 벡터: 높은 유사도 그림과 동일하게 70° */}
                    <line
                      x1="80"
                      y1="205"
                      x2="135"
                      y2="55"
                      className={styles.vectorLineWhite}
                      style={{ strokeWidth: 2 }}
                      markerEnd="url(#arrowWhiteLow)"
                    />

                    {/* 파란색 비교 벡터: 20° */}
                    <line
                      x1="80"
                      y1="205"
                      x2="230"
                      y2="150"
                      className={styles.vectorLineBlue}
                      style={{ strokeWidth: 2 }}
                      markerEnd="url(#arrowBlueLow)"
                    />

                    {/* 두 벡터 사이 약 50° */}
                    <path
                      d="M 136.4 184.5 A 60 60 0 0 0 100.5 148.6"
                      className={styles.angleArc}
                    />

                    <text x="128" y="158" className={styles.angleText}>
                      θ
                    </text>

                    <text
                      x="95"
                      y="45"
                      className={styles.vectorLabelWhite}
                    >
                      vector B
                    </text>

                    <text
                      x="240"
                      y="153"
                      className={styles.vectorLabelBlue}
                    >
                      vector A
                    </text>
                  </svg>
                </div>

                <div className={styles.cosineResult}>
                  <span>큰 θ</span>
                  <strong>cos θ ↓</strong>
                  <p>의미적 방향 차이가 커짐</p>
                </div>
              </div>
            </div>
          </div>
        </article>

        {/* From theory to Word Orbit */}
        <article className={styles.section}>
          <div className={styles.sectionBody}>
            <span className={styles.sectionLabel}>
              FROM THEORY TO WORD ORBIT
            </span>

            <h2>학습된 의미 공간을 게임으로 탐색하기</h2>

            <p className={styles.description}>
              Word Orbit은 플레이 중에 새로운 임베딩 모델을 학습시키는
              서비스가 아닙니다. 대신
              <strong>
                {" "}
                이미 학습된 임베딩 공간을 사용자가 직접 탐색하도록 만든 게임
              </strong>
              입니다.
            </p>

            <div className={styles.flow}>
              <div>
                <span>INPUT</span>
                <strong>추측 단어</strong>
                <p>사용자가 정답일 것 같은 단어를 입력합니다.</p>
              </div>

              <span className={styles.arrow}>→</span>

              <div>
                <span>VECTOR</span>
                <strong>Embedding</strong>
                <p>입력 단어와 정답 단어를 임베딩 벡터로 표현합니다.</p>
              </div>

              <span className={styles.arrow}>→</span>

              <div>
                <span>SIMILARITY</span>
                <strong>Cosine</strong>
                <p>두 벡터의 코사인 유사도를 계산합니다.</p>
              </div>

              <span className={styles.arrow}>→</span>

              <div>
                <span>RANK</span>
                <strong>의미적 순위</strong>
                <p>
                  전체 후보 단어 가운데 정답에 얼마나 가까운지를 순위로
                  보여줍니다.
                </p>
              </div>
            </div>

            <p className={styles.description}>
              사용자가 단어를 입력하면 해당 단어와 정답 단어의 임베딩 벡터를
              비교합니다. 코사인 유사도가 높을수록 의미 공간에서 정답과
              비슷한 방향에 있는 단어로 판단할 수 있습니다.
            </p>

            <p className={styles.description}>
              Word Orbit은 이 결과를 단순한 숫자로만 보여주지 않고,
              <strong>
                {" "}
                순위와 별의 위치·밝기
              </strong>
              로 함께 표현하여 사용자가 임베딩 공간을 탐색하는 과정을
              시각적으로 경험할 수 있도록 구성했습니다. 따라서 메인 화면의
              우주는 단순한 장식이 아니라 단어들이 위치하는 의미 공간을
              시각적으로 해석한 것입니다.
            </p>
          </div>
        </article>

        {/* Human-constrained search research */}
        <article className={styles.section}>
        <div className={styles.sectionBody}>
            <span className={styles.sectionLabel}>
            HUMAN-CONSTRAINED SEARCH
            </span>

            <h2>더 적은 추측으로 정답을 찾을 수 있을까?</h2>

            <p className={styles.description}>
            Word Orbit을 단순한 단어 맞히기 게임이 아니라,
            <strong> 제한된 정보만으로 목표를 찾아가는 탐색 문제</strong>로
            바라보았습니다.
            </p>

            <blockquote className={styles.question}>
            “게임 내부 의미 공간에 직접 접근하지 않고,
            플레이어에게 제공되는 피드백만을 이용할 때
            어떤 탐색 전략이 Word Orbit의 평균 추측 횟수를 최소화할까?”
            </blockquote>

            <p className={styles.description}>
            여기서 중요한 것은 정답의 임베딩 벡터나 전체 단어의 유사도 배열을
            알고 있는 최적 Solver를 만드는 것이 아닙니다.
            실제 플레이어가 게임을 하며 얻을 수 있는
            <strong> 추측 단어, 유사도, 순위와 같은 피드백</strong>만을 이용해
            다음 단어를 선택하는 전략을 비교하는 것이 연구의 목표입니다.
            </p>

            <div className={styles.constraintGrid}>
            <div className={styles.constraintCard}>
                <span>01 · FEEDBACK ONLY</span>
                <strong>관측한 정보만 사용</strong>
                <p>
                현재까지 입력한 단어와 그 단어에 대해 반환된 유사도·순위만
                다음 탐색의 근거로 사용합니다.
                </p>
            </div>

            <div className={styles.constraintCard}>
                <span>02 · LIMITED RANK</span>
                <strong>순위 정보 제한</strong>
                <p>
                전체 vocabulary의 정확한 순위를 그대로 사용하지 않고,
                실제 플레이 환경에 가까운 제한된 순위 정보만 사용합니다.
                </p>
            </div>

            <div className={styles.constraintCard}>
                <span>03 · CANDIDATE BUDGET</span>
                <strong>후보 탐색 폭 제한</strong>
                <p>
                매 턴 수만 개의 단어를 전수 탐색하지 않고,
                한 번에 검토할 수 있는 후보 수를 제한합니다.
                </p>
            </div>
            </div>

            <div className={styles.researchDivider}>
            <span>SEARCH STRATEGIES</span>
            <strong>어떤 방식으로 다음 단어를 선택할까?</strong>
            </div>

            <div className={styles.strategyGrid}>
            <div className={styles.strategyCard}>
                <span className={styles.strategyTag}>GREEDY</span>
                <h3>가장 가까운 곳을 따라가기</h3>
                <p>
                지금까지 가장 높은 유사도를 얻은 단어 주변에서
                연관된 단어를 계속 탐색합니다.
                </p>
                <small>
                직관적이지만 특정 의미 영역의 local optimum에 갇힐 수 있습니다.
                </small>
            </div>

            <div className={styles.strategyCard}>
                <span className={styles.strategyTag}>BROAD PROBE</span>
                <h3>먼저 넓게 탐색하기</h3>
                <p>
                사람·장소·음식·감정·행동처럼 서로 다른 의미 영역의
                기준 단어를 먼저 입력합니다.
                </p>
                <small>
                초반 추측을 탐색에 사용해 정답이 위치한 의미 영역을 찾습니다.
                </small>
            </div>

            <div className={styles.strategyCard}>
                <span className={styles.strategyTag}>CONTRASTIVE</span>
                <h3>비슷하지만 다른 방향 비교하기</h3>
                <p>
                현재 좋은 단어와 관련되면서도 서로 다른 의미 방향을 가진
                단어들을 비교해 다음 이동 방향을 판단합니다.
                </p>
                <small>
                한 의미 영역 안에서 어느 방향으로 더 이동할지를 판별합니다.
                </small>
            </div>

            <div className={styles.strategyCard}>
                <span className={styles.strategyTag}>RANK-BASED</span>
                <h3>상대적인 위치를 이용하기</h3>
                <p>
                단순히 유사도 숫자 하나만 보는 대신 여러 추측의 상대적인
                순서를 비교해 정답이 있을 가능성이 높은 영역을 좁힙니다.
                </p>
                <small>
                여러 기준점으로 의미 공간을 삼각측량하는 방식과 비슷합니다.
                </small>
            </div>
            </div>

            <div className={styles.hybridResearch}>
            <div className={styles.hybridHeader}>
                <span>MAIN RESEARCH CANDIDATE</span>
                <h3>Hybrid Search</h3>
                <p>
                하나의 전략만 반복하지 않고 탐색 상황에 따라
                여러 전략을 전환하는 방식입니다.
                </p>
            </div>

            <div className={styles.hybridFlow}>
                <div className={styles.hybridStep}>
                <span>01 · EXPLORE</span>
                <strong>넓게 탐색</strong>
                <p>
                    서로 다른 의미 영역을 탐색해 정답이 존재할 가능성이 높은
                    영역을 찾습니다.
                </p>
                </div>

                <div className={styles.hybridArrow}>↓</div>

                <div className={styles.hybridStep}>
                <span>02 · DISCRIMINATE</span>
                <strong>방향 판별</strong>
                <p>
                    관련된 후보들을 비교하여 어느 의미 방향으로 이동해야 하는지
                    판단합니다.
                </p>
                </div>

                <div className={styles.hybridArrow}>↓</div>

                <div className={styles.hybridStep}>
                <span>03 · EXPLOIT</span>
                <strong>집중 탐색</strong>
                <p>
                    충분히 가까워졌다면 가장 가능성이 높은 의미 영역을
                    집중적으로 탐색합니다.
                </p>
                </div>
            </div>

            <p className={styles.hybridReturn}>
                탐색이 일정 시간 개선되지 않으면 다시 이전 단계로 돌아가
                다른 의미 방향을 탐색합니다.
            </p>
            </div>

            <div className={styles.researchDivider}>
            <span>EVALUATION</span>
            <strong>어떤 전략이 더 좋은지는 어떻게 판단할까?</strong>
            </div>

            <div className={styles.metricGrid}>
            <div className={styles.metricCard}>
                <span>PRIMARY</span>
                <strong>Success@20</strong>
                <p>20회 이내에 정답을 찾은 게임의 비율</p>
            </div>

            <div className={styles.metricCard}>
                <span>CO-PRIMARY</span>
                <strong>RMG@100</strong>
                <p>최대 100회의 제한을 고려한 평균 추측 횟수</p>
            </div>

            <div className={styles.metricCard}>
                <span>SEARCH</span>
                <strong>Rank ≤ 1000</strong>
                <p>정답과 가까운 의미 영역에 진입하기까지 필요한 추측 수</p>
            </div>
            </div>

            <div className={styles.formulaBox}>
            <span>RESTRICTED MEAN GUESSES</span>

            <strong>RMG@100 = E[min(T, 100)]</strong>

            <p>
                제한된 횟수 안에 정답을 찾지 못한 경우까지 포함하여
                전략의 평균적인 탐색 비용을 비교합니다.
            </p>
            </div>

            <div className={styles.researchStatus}>
            <span>RESEARCH STATUS</span>

            <p>
                현재 단계에서는 특정 전략이 최종적으로 가장 우수하다고
                결론 내린 것이 아니라,
                <strong>
                {" "}
                Greedy · Broad Probe · Contrastive · Rank-based · Hybrid
                </strong>
                전략을 동일한 제한정보 환경에서 비교하기 위한 실험 구조를
                설계했습니다.
            </p>

            <p>
                특히 <strong>Hybrid Search</strong>를 주요 연구 후보로 두고,
                단순 Greedy 전략 및 정보 자체를 활용하지 않는 기준 전략과 비교하여
                실제 추측 횟수를 얼마나 줄일 수 있는지 검증할 예정입니다.
            </p>
            </div>
        </div>
        </article>

        {/* Data source */}
        <article className={styles.section}>
            <div className={styles.sectionBody}>
                <span className={styles.sectionLabel}>
                    DATA SOURCE
                </span>

                <h2>Word Orbit의 단어 데이터</h2>

                <p className={styles.description}>
                    Word Orbit의 단어 데이터는
                    <strong> 위키백과의 한국어 텍스트를 기반</strong>으로 구성했습니다.
                    위키백과에서 수집한 대규모 한국어 텍스트를 바탕으로 게임에서 탐색할
                    수 있는 단어 후보를 만들고, 프로젝트에 적합한 형태로 전처리하여
                    단어장을 구성했습니다.
                </p>

                <div className={styles.sourceCard}>
                    <div className={styles.sourceIcon}>W</div>

                <div>
                    <span>BASE VOCABULARY</span>

                    <h3>위키낱말사전(Wiktionary)</h3>

                    <p>
                        Word Orbit에서 사용하는 전체 단어 공간의 기반은
                        <strong> 위키낱말사전(Wiktionary) </strong>입니다.
                        위키백과에 등장하는 다양한 한국어 어휘를 추출하고 정제하여,
                        사용자가 게임에서 입력하고 탐색할 수 있는 59,582개의 단어 후보를
                        구성했습니다.
                    </p>

                    <p className={styles.sourceDetail}>
                        이렇게 구성된 단어들은 임베딩 모델의 의미 공간에서 서로 다른
                        위치를 가지며, 사용자가 입력한 단어와 정답 단어 사이의 코사인
                        유사도와 순위를 계산하는 데 활용됩니다.
                    </p>
                </div>
            </div>

            <div className={styles.sourceCard}>
                <div className={styles.sourceIcon}>가</div>

                <div>
                    <span>ANSWER FILTERING</span>

                    <h3>국립국어원 언어 자료</h3>

                    <p>
                        전체 단어가 모두 게임의 정답으로 적합한 것은 아니기 때문에,
                        <strong>
                            {" "}
                            국립국어원 말뭉치의 신문·대화·온라인 자료를 활용하여 
                            정답 후보 단어의 실제 사용 빈도와 
                            여러 장르에서의 사용 안정성
                        </strong>
                        을 검증했습니다.
                        이후 단어의 metadata 조건과 
                        genre frequency 조건을 
                        함께 적용하여 최종 정답 후보군 4,785개를 구성했습니다.
                    </p>

                    <p className={styles.sourceDetail}>
                        이를 통해 지나치게 특수하거나 게임 정답으로 사용하기 어려운
                        단어를 걸러내고, 사용자가 의미를 추론하기에 적절한 단어들을
                        정답 후보군으로 구성했습니다.
                    </p>
                </div>
            </div>

            <p className={styles.sourceNotice}>
                단어 데이터 기반: 위키낱말사전(Wiktionary) · 정답 후보 필터링 자료: 국립국어원
            </p>
        </div>
        </article>
      </section>

      <section className={styles.finalSection}>
        <span>READY TO EXPLORE?</span>

        <h2>
          이제 직접 단어 우주를
          <br />
          탐험해보세요.
        </h2>

        <Link href="/" className={styles.playButton}>
          Word Orbit 시작하기
          <span>→</span>
        </Link>
      </section>

      <footer className={styles.footer}>
        <div>
          <span>✦</span>
          WORD ORBIT
        </div>

        <p>
          Word Embedding based Semantic Word Exploration
        </p>
      </footer>
    </main>
  );
}
