"use client";

import { Canvas } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";

function SceneContent() {
  return (
    <>
      <ambientLight intensity={1.6} />

      <axesHelper args={[5]} />

      <Grid
        args={[10, 10]}
        position={[0, -0.01, 0]}
        cellSize={1}
        cellThickness={0.6}
        cellColor="#cbd5e1"
        sectionSize={5}
        sectionThickness={1}
        sectionColor="#94a3b8"
        fadeDistance={15}
        fadeStrength={1}
        infiniteGrid
      />

      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.08}
        enablePan
        enableRotate
        enableZoom
        minDistance={4}
        maxDistance={18}
      />
    </>
  );
}

export function EmbeddingScene() {
  return (
    <div className="h-full w-full">
      <Canvas
        camera={{
          position: [7, 6, 7],
          fov: 45,
          near: 0.1,
          far: 100,
        }}
        dpr={[1, 2]}
      >
        <color attach="background" args={["#f8fafc"]} />
        <SceneContent />
      </Canvas>
    </div>
  );
}