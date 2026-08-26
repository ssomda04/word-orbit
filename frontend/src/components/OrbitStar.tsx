"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

type OrbitStarProps = {
  position: [number, number, number];
  color?: string;
  size?: number;
  selected?: boolean;
};

function createStarTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;

  const ctx = canvas.getContext("2d");
  if (!ctx) return new THREE.CanvasTexture(canvas);

  const center = 128;

  ctx.clearRect(0, 0, 256, 256);

  // --------------------------------
  // 1. 바깥쪽 은은한 빛 번짐
  // --------------------------------
  const glow = ctx.createRadialGradient(
    center,
    center,
    0,
    center,
    center,
    70
  );

  glow.addColorStop(0, "rgba(255,255,255,1)");
  glow.addColorStop(0.08, "rgba(255,255,255,0.95)");
  glow.addColorStop(0.22, "rgba(255,255,255,0.35)");
  glow.addColorStop(0.5, "rgba(255,255,255,0.10)");
  glow.addColorStop(1, "rgba(255,255,255,0)");

  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(center, center, 70, 0, Math.PI * 2);
  ctx.fill();

  // --------------------------------
  // 2. 가로 십자가 빛
  // --------------------------------
  const horizontal = ctx.createLinearGradient(0, center, 256, center);

  horizontal.addColorStop(0, "rgba(255,255,255,0)");
  horizontal.addColorStop(0.32, "rgba(255,255,255,0.03)");
  horizontal.addColorStop(0.44, "rgba(255,255,255,0.20)");
  horizontal.addColorStop(0.5, "rgba(255,255,255,1)");
  horizontal.addColorStop(0.56, "rgba(255,255,255,0.20)");
  horizontal.addColorStop(0.68, "rgba(255,255,255,0.03)");
  horizontal.addColorStop(1, "rgba(255,255,255,0)");

  ctx.fillStyle = horizontal;
  ctx.fillRect(0, center - 1.5, 256, 3);

  // --------------------------------
  // 3. 세로 십자가 빛
  // --------------------------------
  const vertical = ctx.createLinearGradient(center, 0, center, 256);

  vertical.addColorStop(0, "rgba(255,255,255,0)");
  vertical.addColorStop(0.32, "rgba(255,255,255,0.03)");
  vertical.addColorStop(0.44, "rgba(255,255,255,0.20)");
  vertical.addColorStop(0.5, "rgba(255,255,255,1)");
  vertical.addColorStop(0.56, "rgba(255,255,255,0.20)");
  vertical.addColorStop(0.68, "rgba(255,255,255,0.03)");
  vertical.addColorStop(1, "rgba(255,255,255,0)");

  ctx.fillStyle = vertical;
  ctx.fillRect(center - 1.5, 0, 3, 256);

  // --------------------------------
  // 4. 별 중심
  // --------------------------------
  const core = ctx.createRadialGradient(
    center,
    center,
    0,
    center,
    center,
    11
  );

  core.addColorStop(0, "rgba(255,255,255,1)");
  core.addColorStop(0.35, "rgba(255,255,255,1)");
  core.addColorStop(0.7, "rgba(255,255,255,0.7)");
  core.addColorStop(1, "rgba(255,255,255,0)");

  ctx.fillStyle = core;
  ctx.beginPath();
  ctx.arc(center, center, 11, 0, Math.PI * 2);
  ctx.fill();

  const texture = new THREE.CanvasTexture(canvas);

  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;

  return texture;
}

export default function OrbitStar({
  position,
  color = "#ffffff",
  size = 0.28,
  selected = false,
}: OrbitStarProps) {
  const starRef = useRef<THREE.Sprite>(null);

  const texture = useMemo(() => createStarTexture(), []);

  useFrame(({ clock }) => {
    if (!starRef.current) return;

    // 거의 눈에 안 띌 정도의 은은한 반짝임
    const offset =
      position[0] * 0.7 +
      position[1] * 0.4 +
      position[2] * 0.2;

    const pulse =
      1 +
      Math.sin(clock.elapsedTime * 1.5 + offset) *
        (selected ? 0.045 : 0.025);

    const baseSize = selected ? size * 1.15 : size;

    starRef.current.scale.set(
      baseSize * pulse,
      baseSize * pulse,
      1
    );
  });

  return (
    <sprite
      ref={starRef}
      position={position}
      scale={[size, size, 1]}
    >
      <spriteMaterial
        map={texture}
        color={color}
        transparent
        opacity={selected ? 1 : 0.92}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
        toneMapped={false}
      />
    </sprite>
  );
}