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

              <strong>
                sim(a, b) = (a · b) / (‖a‖ ‖b‖)
              </strong>

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

            <div className={styles.similarityExample}>
              <div>
                <span>비슷한 의미의 단어</span>
                <strong>방향이 비슷함</strong>

                <div className={styles.bar}>
                  <span style={{ width: "88%" }} />
                </div>
              </div>

              <div>
                <span>관련성이 낮은 단어</span>
                <strong>방향 차이가 큼</strong>

                <div className={styles.bar}>
                  <span style={{ width: "28%" }} />
                </div>
              </div>
            </div>

            <p className={styles.note}>
              ※ 위 막대는 코사인 유사도의 의미를 설명하기 위한 시각적
              예시이며, 실제 단어의 유사도는 사용한 임베딩 모델에 따라 달라질
              수 있습니다.
            </p>
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

        {/* Search / optimization */}
        <article className={styles.section}>
          <div className={styles.sectionBody}>
            <span className={styles.sectionLabel}>
              SEARCH &amp; OPTIMIZATION
            </span>

            <h2>더 적은 추측으로 정답을 찾을 수 있을까?</h2>

            <p className={styles.description}>
              여기서 한 단계 더 나아가면 Word Orbit은 하나의
              <strong> 탐색 문제(Search Problem)</strong>로 볼 수 있습니다.
            </p>

            <blockquote className={styles.question}>
              “이미 얻은 유사도 정보를 이용하면 다음 추측을 더 효율적으로
              선택할 수 있지 않을까?”
            </blockquote>

            <p className={styles.description}>
              사용자가 어떤 단어 gᵢ를 추측하고 정답과의 유사도 sᵢ를 얻었다고
              하면, 이 값은 단순히 이번 추측이 가까웠는지 멀었는지를
              알려주는 점수만은 아닙니다. 지금까지 얻은 여러 유사도 값을
              이용하면
              <strong>
                {" "}
                정답 후보가 만족해야 하는 조건
              </strong>
              을 만들 수 있습니다.
            </p>

            <div className={styles.formulaBox}>
              <span>CANDIDATE SPACE</span>

              <strong>
                Cₜ = {"{"} w ∈ V : |sim(gᵢ, w) − sᵢ| ≤ ε, ∀ i = 1,...,t {"}"}
              </strong>

              <p>
                지금까지의 모든 추측에서 얻은 유사도 조건을 동시에 만족하는
                단어만 후보로 남기면, 추측이 반복될수록 가능한 정답 후보
                공간을 점차 줄여갈 수 있습니다.
              </p>
            </div>

            <div className={styles.strategy}>
              <div>
                <span>EXPLORE</span>
                <strong>넓게 탐색</strong>
                <p>
                  처음에는 서로 다른 의미 영역의 단어를 입력해 정답이 위치한
                  대략적인 영역을 탐색합니다.
                </p>
              </div>

              <div className={styles.strategyArrow}>→</div>

              <div>
                <span>NARROW</span>
                <strong>후보 축소</strong>
                <p>
                  지금까지 얻은 유사도 정보를 이용해 가능성이 낮은 영역을
                  제거합니다.
                </p>
              </div>

              <div className={styles.strategyArrow}>→</div>

              <div>
                <span>EXPLOIT</span>
                <strong>집중 탐색</strong>
                <p>
                  후보가 좁혀진 이후에는 유사도가 높은 의미 영역 주변의
                  단어를 집중적으로 탐색합니다.
                </p>
              </div>
            </div>

            <p className={styles.researchNote}>
              이 최적화 방법은 현재 Word Orbit이 자동으로 정답을 찾아주는
              기능이 아니라, 게임을
              <strong> 임베딩 공간에서의 탐색 문제</strong>로 바라보고 추가로
              탐구한 전략입니다.
            </p>
          </div>
        </article>

        {/* Data source */}
        <article className={styles.section}>
          <div className={styles.sectionBody}>
            <span className={styles.sectionLabel}>
              DATA SOURCE
            </span>

            <h2>Word Orbit의 단어 데이터</h2>

            <div className={styles.sourceCard}>
              <div className={styles.sourceIcon}>가</div>

              <div>
                <span>VOCABULARY SOURCE</span>

                <h3>국립국어원 모두의 말뭉치</h3>

                <p>
                  Word Orbit에서 사용하는 한국어 단어 목록은 국립국어원
                  언어정보나눔터의
                  <strong> 「모두의 말뭉치」</strong>에서 제공하는 한국어 언어
                  자료를 바탕으로 구성했습니다.
                </p>

                <p className={styles.sourceDetail}>
                  원본 언어 자료에서 프로젝트에 활용할 단어를 추출하고,
                  게임에 적합한 형태로 전처리 및 단어 선별 과정을 거쳐 Word
                  Orbit의 단어장을 구성했습니다.
                </p>
              </div>
            </div>

            <p className={styles.sourceNotice}>
              출처: 국립국어원 언어정보나눔터 「모두의 말뭉치」
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
