"use client";

import { useRef } from "react";
import * as THREE from "three";

import {
  Bloom,
  EffectComposer,
} from "@react-three/postprocessing";

import {
  Canvas,
  useFrame,
} from "@react-three/fiber";

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
const SPECTRAL_COLORS: Record<
  SpectralType,
  string
> = {
  M: "#ff4d5d",
  K: "#ff9838",
  G: "#ffd84d",
  F: "#fff1a8",
  A: "#ffffff",
  B: "#b8e1ff",
  O: "#5dbdff",
};

/*
 * DOM Canvas를 사용하지 않고
 * Three.js DataTexture로 별 모양을 직접 생성한다.
 *
 * 중심:
 *   밝은 작은 점
 *
 * 주변:
 *   은은한 원형 Glow
 *
 * 가로/세로:
 *   오래된 UI에서 보였던 십자가 형태의 빛줄기
 */
function createStarTexture(): THREE.DataTexture {
  const textureSize = 256;

  const data = new Uint8Array(
    textureSize * textureSize * 4,
  );

  const center =
    (textureSize - 1) / 2;

  for (
    let y = 0;
    y < textureSize;
    y += 1
  ) {
    for (
      let x = 0;
      x < textureSize;
      x += 1
    ) {
      /*
       * -1 ~ 1 범위로 좌표 정규화
       */
      const dx =
        (x - center) / center;

      const dy =
        (y - center) / center;

      const radius = Math.sqrt(
        dx * dx + dy * dy,
      );

      /*
       * 별 중심의 매우 밝은 코어
       */
      const core = Math.exp(
        -(radius * radius) * 650,
      );

      /*
       * 중심 주변의 부드러운 빛 번짐
       */
      const halo = Math.exp(
        -(radius * radius) * 22,
      );

      /*
       * 좌우로 길게 뻗는 광선.
       *
       * dy가 0에 가까울수록 밝고,
       * 좌우 끝으로 갈수록 천천히 사라진다.
       */
      const horizontalRay =
        Math.exp(
          -Math.abs(dy) * 115,
        ) *
        Math.exp(
          -Math.abs(dx) * 3.3,
        );

      /*
       * 위아래로 길게 뻗는 광선.
       */
      const verticalRay =
        Math.exp(
          -Math.abs(dx) * 115,
        ) *
        Math.exp(
          -Math.abs(dy) * 3.3,
        );

      /*
       * 모든 발광 효과 결합.
       */
      const alpha = Math.min(
        1,
        core * 1.6 +
          halo * 0.22 +
          horizontalRay * 0.72 +
          verticalRay * 0.72,
      );

      const index =
        (y * textureSize + x) * 4;

      /*
       * 텍스처 자체는 흰색.
       *
       * 실제 별 색상은
       * SpriteMaterial의 color로 입힌다.
       */
      data[index] = 255;
      data[index + 1] = 255;
      data[index + 2] = 255;
      data[index + 3] =
        Math.round(alpha * 255);
    }
  }

  const texture =
    new THREE.DataTexture(
      data,
      textureSize,
      textureSize,
      THREE.RGBAFormat,
    );

  texture.needsUpdate = true;

  texture.minFilter =
    THREE.LinearFilter;

  texture.magFilter =
    THREE.LinearFilter;

  texture.generateMipmaps = false;

  return texture;
}

/*
 * 별보다 훨씬 부드러운 원형 Glow 텍스처.
 */
function createGlowTexture(): THREE.DataTexture {
  const textureSize = 256;

  const data = new Uint8Array(
    textureSize * textureSize * 4,
  );

  const center =
    (textureSize - 1) / 2;

  for (
    let y = 0;
    y < textureSize;
    y += 1
  ) {
    for (
      let x = 0;
      x < textureSize;
      x += 1
    ) {
      const dx =
        (x - center) / center;

      const dy =
        (y - center) / center;

      const radiusSquared =
        dx * dx + dy * dy;

      /*
       * 중심에서 바깥으로 부드럽게 사라지는 Glow
       */
      const alpha = Math.exp(
        -radiusSquared * 5.5,
      );

      const index =
        (y * textureSize + x) * 4;

      data[index] = 255;
      data[index + 1] = 255;
      data[index + 2] = 255;
      data[index + 3] =
        Math.round(alpha * 255);
    }
  }

  const texture =
    new THREE.DataTexture(
      data,
      textureSize,
      textureSize,
      THREE.RGBAFormat,
    );

  texture.needsUpdate = true;

  texture.minFilter =
    THREE.LinearFilter;

  texture.magFilter =
    THREE.LinearFilter;

  texture.generateMipmaps = false;

  return texture;
}

/*
 * 모든 별이 같은 텍스처를 공유한다.
 *
 * Guess가 많아져도 단어마다 Texture를
 * 새로 생성하지 않도록 한다.
 */
const STAR_TEXTURE =
  createStarTexture();

const GLOW_TEXTURE =
  createGlowTexture();

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

interface StarSpriteProps {
  color: string;
  glowColor?: string;
  size: number;
  selected?: boolean;
  phase?: number;
  onClick?: () => void;
}

/*
 * 3D 공간에 위치하지만 항상 카메라를 바라보는 별.
 *
 * Sprite를 사용하므로 카메라를 회전해도
 * 십자가 별 모양이 항상 정면으로 유지된다.
 */
function StarSprite({
  color,
  glowColor = color,
  size,
  selected = false,
  phase = 0,
  onClick,
}: StarSpriteProps) {
  const starRef =
    useRef<THREE.Sprite>(null);

  const glowRef =
    useRef<THREE.Sprite>(null);

  /*
   * 너무 요란하지 않게 아주 약하게만 반짝인다.
   */
  useFrame(({ clock }) => {
    const pulse =
      1 +
      Math.sin(
        clock.elapsedTime * 1.45 +
          phase,
      ) *
        (selected
          ? 0.035
          : 0.018);

    if (starRef.current) {
      const starSize =
        size * pulse;

      starRef.current.scale.set(
        starSize,
        starSize,
        1,
      );
    }

    if (glowRef.current) {
      const glowSize =
        size *
        1.75 *
        pulse;

      glowRef.current.scale.set(
        glowSize,
        glowSize,
        1,
      );
    }
  });

  return (
    <>
      {/*
       * 넓고 부드러운 바깥쪽 발광
       */}
      <sprite
        ref={glowRef}
        scale={[
          size * 1.75,
          size * 1.75,
          1,
        ]}
      >
        <spriteMaterial
          map={GLOW_TEXTURE}
          color={glowColor}
          transparent
          opacity={
            selected
              ? 0.32
              : 0.2
          }
          blending={
            THREE.AdditiveBlending
          }
          depthWrite={false}
          toneMapped={false}
        />
      </sprite>

      {/*
       * 십자가 광선 + 중심 코어
       */}
      <sprite
        ref={starRef}
        scale={[
          size,
          size,
          1,
        ]}
        onClick={(event) => {
          if (!onClick) {
            return;
          }

          event.stopPropagation();
          onClick();
        }}
      >
        <spriteMaterial
          map={STAR_TEXTURE}
          color={color}
          transparent
          opacity={1}
          alphaTest={0.003}
          blending={
            THREE.AdditiveBlending
          }
          depthWrite={false}
          toneMapped={false}
        />
      </sprite>
    </>
  );
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
  const position =
    getPosition(
      guess,
      index,
    );

  const color =
    SPECTRAL_COLORS[
      guess.spectralType
    ];

  /*
   * 이전 버전처럼 별 자체는 작게 유지.
   *
   * 텍스처 전체 크기 안에서
   * 중심 코어는 훨씬 작기 때문에
   * 실제 화면에서는 작은 별 + 긴 광선 형태로 보인다.
   */
  const size = selected
    ? 0.74
    : best
      ? 0.66
      : 0.58;

  return (
    <group position={position}>
      {/*
       * 선택된 단어에서
       * 정답 원점으로 이어지는 선
       */}
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
          opacity={0.16}
          lineWidth={1}
        />
      )}

      <StarSprite
        color={color}
        glowColor={color}
        size={size}
        selected={selected}
        phase={index * 0.83}
        onClick={onSelect}
      />

      <Html
        center
        distanceFactor={8}
        style={{
          pointerEvents: "none",
        }}
      >
        <div
          style={{
            transform:
              "translateY(20px)",
            whiteSpace: "nowrap",
            fontSize: selected
              ? "13px"
              : "11px",
            fontWeight: selected
              ? 700
              : 500,
            color: "#f4f8ff",
            textShadow:
              `0 0 7px ${color}`,
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
      <StarSprite
        color="#ffffff"
        glowColor="#8fd0ff"
        size={0.92}
        selected
        phase={1.4}
      />

      <Html
        center
        distanceFactor={8}
        style={{
          pointerEvents: "none",
        }}
      >
        <div
          style={{
            transform:
              "translateY(30px)",
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
        gl={{
          alpha: true,
          antialias: true,
        }}
      >
        {/*
         * 기본 배경은 CSS 우주 배경이 보이도록
         * Canvas를 투명하게 유지.
         */}

        <ambientLight
          intensity={0.25}
        />

        {/*
         * 배경 작은 별.
         *
         * 주인공인 추측 단어 별보다
         * 너무 밝지 않게 유지한다.
         */}
        <Stars
          radius={45}
          depth={30}
          count={1500}
          factor={2.2}
          saturation={0.1}
          fade
          speed={0.1}
        />

        <AnswerStar />

        {guesses.map(
          (guess, index) => (
            <WordStar
              key={guess.guessId}
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

        {/*
         * Sprite 자체에 Glow가 있기 때문에
         * Bloom을 너무 세게 적용하면
         * 예쁜 십자가 모양이 뭉개진다.
         *
         * 기존 1.8에서 조금 낮춰
         * 중심부만 추가로 반짝이도록 조정.
         */}
        <EffectComposer>
          <Bloom
            intensity={1.15}
            luminanceThreshold={0.25}
            luminanceSmoothing={0.9}
            mipmapBlur
          />
        </EffectComposer>

        {/*
         * 마우스 컨트롤
         *
         * 왼쪽 드래그 → 3D 회전
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