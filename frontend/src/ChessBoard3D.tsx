import { useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { ContactShadows, Environment, OrbitControls } from "@react-three/drei";
import { Chess } from "chess.js";
import type { ThreeEvent } from "@react-three/fiber";

type Props = {
  fen: string;
  lastUci?: string | null;
  selected?: string | null;
  legalTargets?: string[];
  interactive?: boolean;
  orientation?: "white" | "black";
  onSquareClick?: (square: string) => void;
};

const FILES = "abcdefgh";
const WHITE = "#f6f1e6";
const BLACK = "#1e2329";
const LIGHT_WOOD = "#e8d5ae";
const DARK_WOOD = "#b07a4a";
const FRAME = "#3a2718";
const SEG = 32;

function sqToPos(sq: string): [number, number, number] {
  const file = FILES.indexOf(sq[0]);
  const rank = Number(sq[1]) - 1;
  return [file - 3.5, 0, 3.5 - rank];
}

function PieceMat({ color, accent = false }: { color: string; accent?: boolean }) {
  return (
    <meshPhysicalMaterial
      color={color}
      roughness={accent ? 0.18 : 0.28}
      metalness={accent ? 0.45 : 0.04}
      clearcoat={accent ? 0.6 : 0.35}
      clearcoatRoughness={0.25}
    />
  );
}

function Base({ color, r = 0.3 }: { color: string; r?: number }) {
  return (
    <group>
      <mesh position={[0, 0.05, 0]}>
        <cylinderGeometry args={[r, r * 1.08, 0.1, SEG]} />
        <PieceMat color={color} />
      </mesh>
      <mesh position={[0, 0.13, 0]}>
        <cylinderGeometry args={[r * 0.72, r * 0.92, 0.08, SEG]} />
        <PieceMat color={color} />
      </mesh>
    </group>
  );
}

function Collar({ y, r, color }: { y: number; r: number; color: string }) {
  return (
    <mesh position={[0, y, 0]}>
      <cylinderGeometry args={[r, r * 0.92, 0.06, SEG]} />
      <PieceMat color={color} />
    </mesh>
  );
}

function PieceMesh({ type, color }: { type: string; color: "w" | "b" }) {
  const mat = color === "w" ? WHITE : BLACK;
  const gold = color === "w" ? "#c9a227" : "#c5c8ce";

  switch (type) {
    case "p":
      return (
        <group>
          <Base color={mat} r={0.26} />
          <mesh position={[0, 0.28, 0]}>
            <cylinderGeometry args={[0.1, 0.16, 0.22, SEG]} />
            <PieceMat color={mat} />
          </mesh>
          <Collar y={0.4} r={0.14} color={mat} />
          <mesh position={[0, 0.54, 0]}>
            <sphereGeometry args={[0.155, SEG, SEG]} />
            <PieceMat color={mat} />
          </mesh>
        </group>
      );
    case "r":
      return (
        <group>
          <Base color={mat} r={0.3} />
          <mesh position={[0, 0.38, 0]}>
            <cylinderGeometry args={[0.18, 0.22, 0.42, SEG]} />
            <PieceMat color={mat} />
          </mesh>
          <mesh position={[0, 0.62, 0]}>
            <cylinderGeometry args={[0.24, 0.24, 0.1, SEG]} />
            <PieceMat color={mat} />
          </mesh>
          {[0, 1, 2, 3].map((i) => {
            const a = (i * Math.PI) / 2 + Math.PI / 4;
            return (
              <mesh key={i} position={[Math.cos(a) * 0.16, 0.72, Math.sin(a) * 0.16]}>
                <boxGeometry args={[0.11, 0.14, 0.11]} />
                <PieceMat color={mat} />
              </mesh>
            );
          })}
        </group>
      );
    case "n":
      return (
        <group>
          <Base color={mat} r={0.28} />
          <mesh position={[0, 0.32, 0.02]} rotation={[0.35, 0.15, 0]}>
            <cylinderGeometry args={[0.1, 0.18, 0.38, SEG]} />
            <PieceMat color={mat} />
          </mesh>
          <mesh position={[0.02, 0.58, 0.08]} rotation={[0.2, 0.35, 0.05]}>
            <sphereGeometry args={[0.18, SEG, SEG]} />
            <PieceMat color={mat} />
          </mesh>
          <mesh position={[0.14, 0.62, 0.18]} rotation={[0.5, 0.4, 0]}>
            <coneGeometry args={[0.1, 0.28, 16]} />
            <PieceMat color={mat} />
          </mesh>
          <mesh position={[-0.04, 0.74, 0.02]} rotation={[0, 0, -0.35]}>
            <coneGeometry args={[0.05, 0.14, 10]} />
            <PieceMat color={mat} />
          </mesh>
        </group>
      );
    case "b":
      return (
        <group>
          <Base color={mat} r={0.28} />
          <mesh position={[0, 0.34, 0]}>
            <cylinderGeometry args={[0.1, 0.16, 0.28, SEG]} />
            <PieceMat color={mat} />
          </mesh>
          <Collar y={0.5} r={0.15} color={mat} />
          <mesh position={[0, 0.72, 0]}>
            <coneGeometry args={[0.16, 0.42, SEG]} />
            <PieceMat color={mat} />
          </mesh>
          <mesh position={[0, 0.96, 0]}>
            <sphereGeometry args={[0.07, 16, 16]} />
            <PieceMat color={gold} accent />
          </mesh>
        </group>
      );
    case "q":
      return (
        <group>
          <Base color={mat} r={0.32} />
          <mesh position={[0, 0.36, 0]}>
            <cylinderGeometry args={[0.12, 0.2, 0.32, SEG]} />
            <PieceMat color={mat} />
          </mesh>
          <Collar y={0.54} r={0.18} color={mat} />
          <mesh position={[0, 0.74, 0]}>
            <sphereGeometry args={[0.2, SEG, SEG]} />
            <PieceMat color={mat} />
          </mesh>
          <mesh position={[0, 0.9, 0]}>
            <cylinderGeometry args={[0.18, 0.16, 0.08, SEG]} />
            <PieceMat color={mat} />
          </mesh>
          {[0, 1, 2, 3, 4].map((i) => {
            const a = (i / 5) * Math.PI * 2;
            return (
              <mesh key={i} position={[Math.cos(a) * 0.14, 1.0, Math.sin(a) * 0.14]}>
                <sphereGeometry args={[0.045, 12, 12]} />
                <PieceMat color={gold} accent />
              </mesh>
            );
          })}
          <mesh position={[0, 1.06, 0]}>
            <sphereGeometry args={[0.06, 12, 12]} />
            <PieceMat color={gold} accent />
          </mesh>
        </group>
      );
    case "k":
      return (
        <group>
          <Base color={mat} r={0.32} />
          <mesh position={[0, 0.38, 0]}>
            <cylinderGeometry args={[0.12, 0.2, 0.36, SEG]} />
            <PieceMat color={mat} />
          </mesh>
          <Collar y={0.58} r={0.18} color={mat} />
          <mesh position={[0, 0.8, 0]}>
            <sphereGeometry args={[0.2, SEG, SEG]} />
            <PieceMat color={mat} />
          </mesh>
          <mesh position={[0, 0.98, 0]}>
            <cylinderGeometry args={[0.1, 0.12, 0.12, SEG]} />
            <PieceMat color={mat} />
          </mesh>
          <mesh position={[0, 1.14, 0]}>
            <boxGeometry args={[0.07, 0.26, 0.07]} />
            <PieceMat color={gold} accent />
          </mesh>
          <mesh position={[0, 1.18, 0]}>
            <boxGeometry args={[0.2, 0.07, 0.07]} />
            <PieceMat color={gold} accent />
          </mesh>
        </group>
      );
    default:
      return null;
  }
}

function BoardScene({
  fen,
  lastUci,
  selected,
  legalTargets,
  interactive,
  orientation,
  onSquareClick,
}: Props) {
  const chess = useMemo(() => new Chess(fen), [fen]);
  const from = lastUci?.slice(0, 2);
  const to = lastUci?.slice(2, 4);
  const legalSet = useMemo(() => new Set(legalTargets), [legalTargets]);
  const flipped = orientation === "black";

  const squares: string[] = [];
  for (let rank = 1; rank <= 8; rank += 1) {
    for (const file of FILES) squares.push(`${file}${rank}`);
  }

  function handleClick(sq: string, event: ThreeEvent<MouseEvent>) {
    event.stopPropagation();
    if (!interactive) return;
    onSquareClick?.(sq);
  }

  return (
    <group rotation={[0, flipped ? Math.PI : 0, 0]}>
      <mesh position={[0, -0.16, 0]}>
        <boxGeometry args={[9.15, 0.28, 9.15]} />
        <meshStandardMaterial color={FRAME} roughness={0.55} />
      </mesh>

      {squares.map((sq) => {
        const [x, , z] = sqToPos(sq);
        const file = FILES.indexOf(sq[0]);
        const rank = Number(sq[1]);
        const dark = (file + rank) % 2 === 0;
        const piece = chess.get(sq as "a1");
        const isLegal = legalSet.has(sq);
        const isCapture = isLegal && Boolean(piece);
        let tile = dark ? DARK_WOOD : LIGHT_WOOD;
        if (sq === selected) tile = "#e2b84a";
        else if (from === sq || to === sq) tile = dark ? "#8f6a3e" : "#d7c48a";
        else if (isCapture) tile = "#c45c48";
        else if (isLegal) tile = dark ? "#9a7a4c" : "#d4c392";

        return (
          <group key={sq} position={[x, 0, z]}>
            <mesh
              position={[0, -0.02, 0]}
              onClick={(e) => handleClick(sq, e)}
              onPointerOver={(e) => {
                e.stopPropagation();
                if (interactive) document.body.style.cursor = "pointer";
              }}
              onPointerOut={() => {
                document.body.style.cursor = "auto";
              }}
            >
              <boxGeometry args={[0.98, 0.08, 0.98]} />
              <meshStandardMaterial color={tile} roughness={0.62} metalness={0.02} />
            </mesh>
            {isLegal && !piece ? (
              <mesh position={[0, 0.06, 0]}>
                <cylinderGeometry args={[0.13, 0.13, 0.03, 20]} />
                <meshStandardMaterial color="#1a7a62" transparent opacity={0.85} />
              </mesh>
            ) : null}
            {piece ? (
              <group onClick={(e) => handleClick(sq, e)}>
                <PieceMesh type={piece.type} color={piece.color} />
              </group>
            ) : null}
          </group>
        );
      })}
    </group>
  );
}

export function ChessBoard3D(props: Props) {
  return (
    <div className={`board board-3d ${props.interactive ? "board-interactive" : ""}`} aria-label="3D chess board">
      <Canvas camera={{ position: [0, 8.6, 8.8], fov: 36 }} dpr={[1, 2]} gl={{ antialias: true }}>
        <color attach="background" args={["#cfdbe8"]} />
        <hemisphereLight args={["#fff6e8", "#8aa0b5", 0.85]} />
        <directionalLight position={[5, 10, 6]} intensity={1.35} />
        <directionalLight position={[-6, 4, -4]} intensity={0.28} />
        <Environment preset="studio" environmentIntensity={0.35} />
        <BoardScene {...props} />
        <ContactShadows position={[0, -0.3, 0]} opacity={0.35} scale={12} blur={2.2} far={6} />
        <OrbitControls
          enablePan={false}
          minDistance={8}
          maxDistance={15}
          minPolarAngle={0.5}
          maxPolarAngle={1.12}
          target={[0, 0.2, 0]}
        />
      </Canvas>
    </div>
  );
}
