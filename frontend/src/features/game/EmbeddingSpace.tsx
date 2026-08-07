"use client";

import * as THREE from "three";

import {
  Bloom,
  EffectComposer,
} from "@react-three/postprocessing";

import { Canvas } from "@react-three/fiber";
import {
  Html,
  Line,
  OrbitControls,
  Stars,
} from "@react-three/drei";

import type { Guess as ApiGuess } from "@/types/api";

export type SpectralType =
  | "M"
  | "K"
  | "G"
  | "F"
  | "A"
  | "B"
  | "O";

export type DisplayGuess = ApiGuess & {
  spectralType: SpectralType;
  similarityPercent: number;
};

interface EmbeddingSpaceProps {
  guesses: DisplayGuess[];
  selectedGuess: DisplayGuess | null;
  bestGuess: DisplayGuess | null;
  onSelectGuess: (guess: DisplayGuess) => void;
}

/*
 * 실제 천체 분광형 느낌을 살린 색상.
 */
const SPECTRAL_COLORS: Record<SpectralType, string> = {
    M: "#ff4d5d",
    K: "#ff9838",
    G: "#ffd84d",
    F: "#fff1a8",
    A: "#ffffff",
    B: "#b8e1ff",
    O: "#5dbdff",
};

/*
 * 순위를 3D 공간에서 정답 별과의 거리로 변환한다.
 *
 * 순위가 높을수록 중심에 가까움.
 */
function interpolate(
  value: number,
  inputStart: number,
  inputEnd: number,
  outputStart: number,
  outputEnd: number,
): number {
  const progress =
    (value - inputStart) /
    (inputEnd - inputStart);

  return (
    outputStart +
    progress *
      (outputEnd - outputStart)
  );
}

function getRadiusByRank(
  rank: number | null,
): number {
  /*
   * rank 구현 전에는 가장 바깥쪽.
   */
  if (rank === null) {
    return 7.2;
  }

  /*
   * O형: 1~10위
   */
  if (rank <= 10) {
    return interpolate(
      rank,
      1,
      10,
      1.1,
      1.8,
    );
  }

  /*
   * B형: 11~50위
   */
  if (rank <= 50) {
    return interpolate(
      rank,
      11,
      50,
      2.0,
      2.7,
    );
  }

  /*
   * A형: 51~150위
   */
  if (rank <= 150) {
    return interpolate(
      rank,
      51,
      150,
      2.9,
      3.7,
    );
  }

  /*
   * F형: 151~350위
   */
  if (rank <= 350) {
    return interpolate(
      rank,
      151,
      350,
      3.9,
      4.7,
    );
  }

  /*
   * G형: 351~650위
   */
  if (rank <= 650) {
    return interpolate(
      rank,
      351,
      650,
      4.9,
      5.6,
    );
  }

  /*
   * K형: 651~1000위
   */
  if (rank <= 1000) {
    return interpolate(
      rank,
      651,
      1000,
      5.8,
      6.5,
    );
  }

  /*
   * M형: 1001위 밖
   */
  return 7.1;
}

/*
 * 같은 순위에 있는 별들이 겹치지 않도록
 * 각 추측 단어에 서로 다른 3차원 방향을 부여한다.
 *
 * 거리에는 의미(rank)가 있지만,
 * 현재 방향 자체에는 임베딩 의미가 없다.
 *
 * 나중에 coordinate {x,y,z}가 구현되면
 * 이 부분만 실제 좌표로 교체하면 된다.
 */
function getDirection(
  index: number,
): [number, number, number] {
  const goldenAngle =
    Math.PI * (3 - Math.sqrt(5));

  const theta =
    index * goldenAngle;

  /*
   * -1 ~ 1 사이 값을 결정적으로 생성.
   */
  const y =
    ((index * 0.754877666) % 1) *
      2 -
    1;

  const horizontalRadius =
    Math.sqrt(
      Math.max(
        0,
        1 - y * y,
      ),
    );

  return [
    Math.cos(theta) *
      horizontalRadius,
    y,
    Math.sin(theta) *
      horizontalRadius,
  ];
}

function getPosition(
  guess: DisplayGuess,
  index: number,
): [number, number, number] {
  /*
   * 향후 실제 coordinate 구현 시에는
   * 아래와 같이 교체 가능하다.
   *
   * if (guess.coordinate) {
   *   return [
   *     guess.coordinate.x,
   *     guess.coordinate.y,
   *     guess.coordinate.z,
   *   ];
   * }
   */

  const radius =
    getRadiusByRank(
      guess.rank,
    );

  const [dx, dy, dz] =
    getDirection(index);

  return [
    dx * radius,
    dy * radius,
    dz * radius,
  ];
}

interface WordStarProps {
  guess: DisplayGuess;
  index: number;
  selected: boolean;
  best: boolean;
  onSelect: () => void;
}


function WordStar({
  guess,
  index,
  selected,
  best,
  onSelect,
}: WordStarProps) {
  const position = getPosition(guess, index);

  const color =
    SPECTRAL_COLORS[guess.spectralType];

  const size = selected
    ? 0.18
    : best
      ? 0.16
      : 0.13;

  return (
    <group position={position}>
      {selected && (
        <Line
          points={[
            [0, 0, 0],
            [
              -position[0],
              -position[1],
              -position[2],
            ],
          ]}
          color={color}
          transparent
          opacity={0.2}
          lineWidth={1}
        />
      )}

      {/* 별빛 바깥 Glow */}
      <mesh scale={2.2}>
        <sphereGeometry args={[size, 24, 24]} />

        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.18}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>

      {/* 실제 별 */}
      <mesh
        onClick={(event) => {
          event.stopPropagation();
          onSelect();
        }}
      >
        <sphereGeometry args={[size, 32, 32]} />

        <meshBasicMaterial
          color={color}
          toneMapped={false}
        />
      </mesh>

      {/* 아주 밝은 흰 중심 */}
      <mesh scale={0.4}>
        <sphereGeometry args={[size, 24, 24]} />

        <meshBasicMaterial
          color="#ffffff"
          toneMapped={false}
        />
      </mesh>

      <Html
        center
        distanceFactor={8}
        style={{
          pointerEvents: "none",
        }}
      >
        <div
          style={{
            transform: "translateY(20px)",
            whiteSpace: "nowrap",
            fontSize: selected ? "13px" : "11px",
            fontWeight: selected ? 700 : 500,
            color: "#f4f8ff",
            textShadow: `0 0 7px ${color}`,
          }}
        >
          {guess.word}
        </div>
      </Html>
    </group>
  );
}

/*
 * 정답 별.
 *
 * 정답은 항상 3D 좌표 원점 (0,0,0)에 고정한다.
 */
function AnswerStar() {
  return (
    <group position={[0, 0, 0]}>
      {/* 넓은 푸른 Glow */}
      <mesh scale={2.8}>
        <sphereGeometry args={[0.25, 32, 32]} />

        <meshBasicMaterial
          color="#9dd7ff"
          transparent
          opacity={0.22}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>

      {/* 흰색 중심 */}
      <mesh>
        <sphereGeometry args={[0.26, 32, 32]} />

        <meshBasicMaterial
          color="#ffffff"
          toneMapped={false}
        />
      </mesh>

      {/* 가장 밝은 코어 */}
      <mesh scale={0.4}>
        <sphereGeometry args={[0.26, 24, 24]} />

        <meshBasicMaterial
          color="#ffffff"
          toneMapped={false}
        />
      </mesh>

      <Html
        center
        distanceFactor={8}
        style={{
          pointerEvents: "none",
        }}
      >
        <div
          style={{
            transform: "translateY(30px)",
            whiteSpace: "nowrap",
            color: "#ffffff",
            fontSize: "11px",
            fontWeight: 700,
            textShadow:
              "0 0 8px #8fd0ff, 0 0 20px #8fd0ff",
          }}
        >
          정답
        </div>
      </Html>
    </group>
  );
}

export default function EmbeddingSpace({
  guesses,
  selectedGuess,
  bestGuess,
  onSelectGuess,
}: EmbeddingSpaceProps) {
  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        minHeight: "520px",
      }}
    >
      <Canvas
        camera={{
          position: [
            0,
            2,
            12,
          ],
          fov: 48,
          near: 0.1,
          far: 100,
        }}
      >
        {/*
         * 기본 배경은 CSS 우주 배경이 보이도록
         * Canvas를 투명하게 유지.
         */}

        <ambientLight
          intensity={0.35}
        />

        {/*
         * 배경 작은 별.
         */}
        <Stars
          radius={45}
          depth={30}
          count={1500}
          factor={2.6}
          saturation={0.15}
          fade
          speed={0.12}
/>

        <AnswerStar />

        {guesses.map(
          (guess, index) => (
            <WordStar
              key={
                guess.guessId
              }
              guess={guess}
              index={index}
              selected={
                selectedGuess?.guessId ===
                guess.guessId
              }
              best={
                bestGuess?.guessId ===
                guess.guessId
              }
              onSelect={() =>
                onSelectGuess(
                  guess,
                )
              }
            />
          ),
        )}

        <EffectComposer>
          <Bloom
            intensity={1.8}
            luminanceThreshold={0.15}
            luminanceSmoothing={0.85}
            mipmapBlur
          />
        </EffectComposer>

        {/*
         * 마우스 컨트롤
         *
         * 왼쪽 드래그 → 회전
         * 휠 → 확대/축소
         */}
        <OrbitControls
          makeDefault
          enableRotate
          enableZoom
          enablePan={false}
          minDistance={5}
          maxDistance={20}
          rotateSpeed={0.65}
          zoomSpeed={0.8}
          dampingFactor={0.07}
          enableDamping
        />
      </Canvas>
    </div>
  );
}