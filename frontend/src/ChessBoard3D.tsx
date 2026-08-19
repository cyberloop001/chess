import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { ContactShadows, Environment, OrbitControls, Sparkles, useGLTF } from "@react-three/drei";
import { Chess } from "chess.js";
import type { ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import { clone } from "three/examples/jsm/utils/SkeletonUtils.js";
import { attachWeapons } from "./weapons";
import { assetUrl } from "./api";

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
const LIGHT_SQ = "#eceae4";
const DARK_SQ = "#2b2e35";
const FRAME_EDGE = "#14161c";

const PIECE_FILE: Record<"w" | "b", Record<string, string>> = {
  w: {
    p: assetUrl("pieces/medieval/ivory/p.glb"),
    r: assetUrl("pieces/medieval/ivory/r.glb"),
    n: assetUrl("pieces/medieval/ivory/n.glb"),
    b: assetUrl("pieces/medieval/ivory/b.glb"),
    q: assetUrl("pieces/medieval/ivory/q.glb"),
    k: assetUrl("pieces/medieval/ivory/k.glb"),
  },
  b: {
    p: assetUrl("pieces/medieval/sun/p.glb"),
    r: assetUrl("pieces/medieval/sun/r.glb"),
    n: assetUrl("pieces/medieval/sun/n.glb"),
    b: assetUrl("pieces/medieval/sun/b.glb"),
    q: assetUrl("pieces/medieval/sun/q.glb"),
    k: assetUrl("pieces/medieval/sun/k.glb"),
  },
};

const PIECE_HEIGHT: Record<string, number> = {
  p: 0.78,
  r: 0.99,
  n: 0.98,
  b: 1.0,
  q: 1.0,
  k: 1.12,
};

function sqToPos(sq: string): [number, number, number] {
  const file = FILES.indexOf(sq[0]);
  const rank = Number(sq[1]) - 1;
  return [file - 3.5, 0, 3.5 - rank];
}

type CaptureEvent = {
  id: number;
  square: string;
  type: string;
  color: "w" | "b";
};

type Travel = { from: string; capture: boolean; knight: boolean };

function easeInOut(t: number) {
  return t * t * (3 - 2 * t);
}

function castleRookTravel(from: string, to: string): { square: string; from: string } | null {
  if (from === "e1" && to === "g1") return { square: "f1", from: "h1" };
  if (from === "e1" && to === "c1") return { square: "d1", from: "a1" };
  if (from === "e8" && to === "g8") return { square: "f8", from: "h8" };
  if (from === "e8" && to === "c8") return { square: "d8", from: "a8" };
  return null;
}

function makeWoodTexture(base: string, grain: string, dark = false): THREE.CanvasTexture {
  const size = 512;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, size, size);

  for (let band = 0; band < 36; band += 1) {
    const y = (band / 36) * size;
    ctx.strokeStyle = grain;
    ctx.globalAlpha = dark ? 0.12 + (band % 4) * 0.04 : 0.08 + (band % 4) * 0.03;
    ctx.lineWidth = 1.5 + (band % 3);
    ctx.beginPath();
    ctx.moveTo(0, y);
    for (let x = 0; x <= size; x += 8) {
      ctx.lineTo(x, y + Math.sin(x * 0.03 + band) * 4);
    }
    ctx.stroke();
  }

  ctx.globalAlpha = 1;
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

let frameWoodMap: THREE.CanvasTexture | null = null;

function tintMaterial(mat: THREE.Material, ghost: boolean) {
  const copy = mat.clone();
  copy.transparent = ghost || copy.transparent;
  if (ghost && "opacity" in copy) copy.opacity = 0.85;
  return copy;
}

function fitPiece(root: THREE.Object3D, type: string, color: "w" | "b") {
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());
  const height = Math.max(size.y, 0.0001);
  const scale = PIECE_HEIGHT[type] / height;
  root.scale.setScalar(scale);
  const centre = box.getCenter(new THREE.Vector3());
  root.position.set(-centre.x * scale, -box.min.y * scale, -centre.z * scale);
  root.rotation.y = color === "w" ? Math.PI : 0;
  return { unit: height, baseY: box.min.y };
}

function PieceMesh({ type, color, ghost = false }: { type: string; color: "w" | "b"; ghost?: boolean }) {
  const url = PIECE_FILE[color][type] ?? PIECE_FILE[color].p;
  const { scene } = useGLTF(url);
  const object = useMemo(() => {
    const root = clone(scene);
    root.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (!mesh.isMesh) return;
      mesh.castShadow = !ghost;
      mesh.receiveShadow = !ghost;
      if (Array.isArray(mesh.material)) {
        mesh.material = mesh.material.map((m) => tintMaterial(m, ghost));
      } else if (mesh.material) {
        mesh.material = tintMaterial(mesh.material, ghost);
      }
    });
    const { unit, baseY } = fitPiece(root, type, color);
    if (!ghost) {
      attachWeapons(
        root,
        type as "k" | "q" | "b" | "n" | "r" | "p",
        color,
        unit,
        baseY,
        color === "w" ? "kingdom" : "sun",
      );
    }
    return root;
  }, [scene, type, color, ghost]);

  return <primitive object={object} />;
}

function PieceActor({
  square,
  type,
  color,
  travel,
  selected,
  phase,
  onClick,
}: {
  square: string;
  type: string;
  color: "w" | "b";
  travel?: Travel;
  selected: boolean;
  phase: number;
  onClick?: () => void;
}) {
  const group = useRef<THREE.Group>(null);
  const visual = useRef<THREE.Group>(null);
  const progress = useRef(travel ? 0 : 1);
  const origin = useRef(sqToPos(travel?.from ?? square));
  const capturing = Boolean(travel?.capture);
  const leaping = Boolean(travel?.knight);

  useEffect(() => {
    origin.current = sqToPos(travel?.from ?? square);
    progress.current = travel ? 0 : 1;
  }, [square, travel?.from, travel?.capture, travel?.knight]);

  useFrame((state, dt) => {
    const g = group.current;
    const v = visual.current;
    if (!g || !v) return;

    const dest = sqToPos(square);
    const duration = leaping ? 0.62 : capturing ? 0.55 : 0.42;
    if (progress.current < 1) {
      progress.current = Math.min(1, progress.current + dt / duration);
    }
    const u = easeInOut(progress.current);
    const x = origin.current[0] + (dest[0] - origin.current[0]) * u;
    const z = origin.current[2] + (dest[2] - origin.current[2]) * u;
    const hop = leaping ? Math.sin(Math.PI * u) * 0.62 : 0;
    const stride = progress.current < 1 ? Math.abs(Math.sin(u * Math.PI * 3)) * 0.05 : 0;
    const idleBob = progress.current >= 1 ? Math.sin(state.clock.elapsedTime * 1.55 + phase) * 0.016 : 0;
    const idleSway = progress.current >= 1 ? Math.sin(state.clock.elapsedTime * 1.05 + phase) * 0.035 : 0;
    const walkLean = progress.current < 1 ? Math.sin(u * Math.PI) * 0.08 : 0;
    const strike = capturing && progress.current > 0.55 && progress.current < 0.92 ? Math.sin(((progress.current - 0.55) / 0.37) * Math.PI) * 0.22 : 0;
    const dx = dest[0] - origin.current[0];
    const dz = dest[2] - origin.current[2];
    const len = Math.hypot(dx, dz) || 1;

    g.position.set(x + (dx / len) * strike * 0.15, hop + stride + idleBob, z + (dz / len) * strike * 0.15);
    v.rotation.x = walkLean + strike * 0.25;
    v.rotation.z = idleSway;
    const pop = selected ? 1.06 : 1;
    v.scale.setScalar(pop);
  });

  return (
    <group ref={group} onClick={(e) => { e.stopPropagation(); onClick?.(); }}>
      {selected ? (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.04, 0]}>
          <ringGeometry args={[0.28, 0.34, 28]} />
          <meshBasicMaterial color="#ffb24a" transparent opacity={0.85} depthWrite={false} />
        </mesh>
      ) : (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.03, 0]}>
          <ringGeometry args={[0.22, 0.26, 24]} />
          <meshBasicMaterial color="#d8c49a" transparent opacity={0.22} depthWrite={false} />
        </mesh>
      )}
      <group ref={visual}>
        <PieceMesh type={type} color={color} />
      </group>
    </group>
  );
}

function CaptureBurst({ event, onDone }: { event: CaptureEvent; onDone: () => void }) {
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const flashRef = useRef<THREE.Mesh>(null);
  const ghostRef = useRef<THREE.Group>(null);
  const sparks = useMemo(
    () =>
      Array.from({ length: 48 }, () => ({
        x: (Math.random() - 0.5) * 0.12,
        y: 0.2 + Math.random() * 0.2,
        z: (Math.random() - 0.5) * 0.12,
        vx: (Math.random() - 0.5) * 3.2,
        vy: 1.4 + Math.random() * 2.6,
        vz: (Math.random() - 0.5) * 3.2,
        life: 0.55 + Math.random() * 0.4,
        age: 0,
        s: 0.035 + Math.random() * 0.05,
      })),
    [event.id],
  );
  const elapsed = useRef(0);

  useFrame((_, dt) => {
    elapsed.current += dt;
    const t = elapsed.current;
    const inst = meshRef.current;
    if (inst) {
      sparks.forEach((p, i) => {
        p.age += dt;
        p.vy -= 6.5 * dt;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.z += p.vz * dt;
        const live = Math.max(0, 1 - p.age / p.life);
        dummy.position.set(p.x, p.y, p.z);
        dummy.scale.setScalar(p.s * live);
        dummy.rotation.set(p.age * 6, p.age * 4, 0);
        dummy.updateMatrix();
        inst.setMatrixAt(i, dummy.matrix);
      });
      inst.instanceMatrix.needsUpdate = true;
    }
    if (flashRef.current) {
      const mat = flashRef.current.material as THREE.MeshBasicMaterial;
      mat.opacity = Math.max(0, 0.7 - t * 1.6);
      flashRef.current.scale.setScalar(1 + t * 2.4);
    }
    if (ghostRef.current) {
      ghostRef.current.position.y = -t * 0.7;
      ghostRef.current.rotation.z = t * 1.1;
      ghostRef.current.traverse((child) => {
        const mesh = child as THREE.Mesh;
        if (!mesh.isMesh) return;
        const mat = mesh.material as THREE.MeshPhysicalMaterial;
        if (mat) mat.opacity = Math.max(0, 0.9 - t * 1.4);
      });
    }
    if (t > 0.95) onDone();
  });

  const hue = event.color === "w" ? "#f3e4c4" : "#c45c48";

  return (
    <group>
      <group ref={ghostRef}>
        <PieceMesh type={event.type} color={event.color} ghost />
      </group>
      <mesh ref={flashRef} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.06, 0]}>
        <circleGeometry args={[0.38, 28]} />
        <meshBasicMaterial color="#ffb24a" transparent opacity={0.7} depthWrite={false} />
      </mesh>
      <Sparkles count={28} scale={0.85} size={4} speed={1.8} color="#ff9a3c" opacity={0.9} />
      <instancedMesh ref={meshRef} args={[undefined, undefined, sparks.length]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color={hue} emissive="#ff9a3c" emissiveIntensity={0.55} roughness={0.4} />
      </instancedMesh>
    </group>
  );
}

function SquareCorners() {
  const s = 0.44;
  const len = 0.14;
  const t = 0.016;
  const corners: Array<[number, number]> = [
    [-1, -1],
    [1, -1],
    [-1, 1],
    [1, 1],
  ];
  return (
    <group position={[0, 0.03, 0]}>
      {corners.map(([sx, sz], i) => (
        <group key={i} position={[sx * s, 0, sz * s]}>
          <mesh>
            <boxGeometry args={[len, t, t]} />
            <meshBasicMaterial color="#ffffff" />
          </mesh>
          <mesh>
            <boxGeometry args={[t, t, len]} />
            <meshBasicMaterial color="#ffffff" />
          </mesh>
        </group>
      ))}
    </group>
  );
}

function BoardFrame() {
  frameWoodMap ??= makeWoodTexture("#e2c797", "#b8945e", false);
  return (
    <group>
      <mesh position={[0, -0.18, 0]} receiveShadow>
        <boxGeometry args={[9.45, 0.32, 9.45]} />
        <meshStandardMaterial map={frameWoodMap} color="#d8bc8e" roughness={0.64} metalness={0} />
      </mesh>
      <mesh position={[0, -0.02, 0]} receiveShadow>
        <boxGeometry args={[9.62, 0.06, 9.62]} />
        <meshStandardMaterial color={FRAME_EDGE} roughness={0.75} />
      </mesh>
    </group>
  );
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
  const prevFen = useRef(fen);
  const [capture, setCapture] = useState<CaptureEvent | null>(null);
  const [travelBySquare, setTravelBySquare] = useState<Record<string, Travel>>({});

  useEffect(() => {
    const previous = prevFen.current;
    prevFen.current = fen;
    if (!lastUci || lastUci.length < 4 || previous === fen) {
      return;
    }
    try {
      const before = new Chess(previous);
      const origin = lastUci.slice(0, 2);
      const dest = lastUci.slice(2, 4);
      const move = before.move({
        from: origin,
        to: dest,
        promotion: lastUci[4] as "q" | "r" | "b" | "n" | undefined,
      });
      if (!move) return;
      const next: Record<string, Travel> = {
        [dest]: { from: origin, capture: Boolean(move.captured), knight: move.piece === "n" },
      };
      const rook = castleRookTravel(origin, dest);
      if (rook) next[rook.square] = { from: rook.from, capture: false, knight: false };
      setTravelBySquare(next);

      if (move.captured) {
        const taken = move.captured;
        const square = move.flags.includes("e") ? `${move.to[0]}${move.from[1]}` : move.to;
        const wait = window.setTimeout(() => {
          setCapture({
            id: Date.now(),
            square,
            type: taken,
            color: move.color === "w" ? "b" : "w",
          });
        }, 320);
        return () => window.clearTimeout(wait);
      }
    } catch {
      /* ignore illegal reconstruct */
    }
  }, [fen, lastUci]);

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
      <BoardFrame />

      {squares.map((sq) => {
        const [x, , z] = sqToPos(sq);
        const file = FILES.indexOf(sq[0]);
        const rank = Number(sq[1]);
        const dark = (file + rank) % 2 === 0;
        const piece = chess.get(sq as "a1");
        const isLegal = legalSet.has(sq);
        const isCapture = isLegal && Boolean(piece);
        let tile = dark ? DARK_SQ : LIGHT_SQ;
        if (sq === selected) tile = "#f2d27a";
        else if (from === sq || to === sq) tile = dark ? "#4a4034" : "#ddd2b6";
        else if (isCapture) tile = "#c45c48";
        else if (isLegal) tile = dark ? "#3d4450" : "#d8d4c8";

        return (
          <group key={sq} position={[x, 0, z]}>
            <mesh
              position={[0, -0.02, 0]}
              receiveShadow
              onClick={(e) => handleClick(sq, e)}
              onPointerOver={(e) => {
                e.stopPropagation();
                if (interactive) document.body.style.cursor = "pointer";
              }}
              onPointerOut={() => {
                document.body.style.cursor = "auto";
              }}
            >
              <boxGeometry args={[0.985, 0.08, 0.985]} />
              <meshStandardMaterial color={tile} roughness={0.78} metalness={0} />
            </mesh>
            {sq === selected ? <SquareCorners /> : null}
            {isLegal && !piece ? (
              <mesh position={[0, 0.06, 0]}>
                <cylinderGeometry args={[0.12, 0.12, 0.03, 20]} />
                <meshStandardMaterial color="#ff9a3c" transparent opacity={0.85} />
              </mesh>
            ) : null}
          </group>
        );
      })}
      {squares.map((sq) => {
        const piece = chess.get(sq as "a1");
        if (!piece) return null;
        const [x, , z] = sqToPos(sq);
        return (
          <PieceActor
            key={`${piece.color}${piece.type}-${sq}`}
            square={sq}
            type={piece.type}
            color={piece.color}
            travel={travelBySquare[sq]}
            selected={sq === selected}
            phase={(x + z) * 0.7}
            onClick={() => {
              if (!interactive) return;
              onSquareClick?.(sq);
            }}
          />
        );
      })}
      {capture ? (
        <group position={sqToPos(capture.square)}>
          <CaptureBurst event={capture} onDone={() => setCapture(null)} />
        </group>
      ) : null}
    </group>
  );
}

export function ChessBoard3D(props: Props) {
  return (
    <div className={`board board-3d ${props.interactive ? "board-interactive" : ""}`} aria-label="3D chess board">
      <Canvas
        shadows
        camera={{ position: [0, 8.4, 9.2], fov: 34 }}
        dpr={[1, 2]}
        gl={{ antialias: true }}
      >
        <color attach="background" args={["#2a2e3a"]} />
        <hemisphereLight args={["#fff4e6", "#6a7588", 0.62]} />
        <directionalLight
          position={[-3.5, 10, 7]}
          intensity={1.7}
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
          shadow-camera-far={24}
          shadow-camera-left={-8}
          shadow-camera-right={8}
          shadow-camera-top={8}
          shadow-camera-bottom={-8}
        />
        <directionalLight position={[6, 4, -3]} intensity={0.2} />
        <Environment preset="warehouse" environmentIntensity={0.22} />
        <Suspense fallback={null}>
          <BoardScene {...props} />
        </Suspense>
        <ContactShadows position={[0, -0.34, 0]} opacity={0.38} scale={12} blur={2.6} far={7} />
        <OrbitControls
          enablePan={false}
          minDistance={8}
          maxDistance={15}
          minPolarAngle={0.5}
          maxPolarAngle={1.12}
          target={[0, 0.15, 0]}
        />
      </Canvas>
    </div>
  );
}

useGLTF.preload("/pieces/medieval/ivory/p.glb");
useGLTF.preload("/pieces/medieval/ivory/r.glb");
useGLTF.preload("/pieces/medieval/ivory/n.glb");
useGLTF.preload("/pieces/medieval/ivory/b.glb");
useGLTF.preload("/pieces/medieval/ivory/q.glb");
useGLTF.preload("/pieces/medieval/ivory/k.glb");
useGLTF.preload("/pieces/medieval/sun/p.glb");
useGLTF.preload("/pieces/medieval/sun/r.glb");
useGLTF.preload("/pieces/medieval/sun/n.glb");
useGLTF.preload("/pieces/medieval/sun/b.glb");
useGLTF.preload("/pieces/medieval/sun/q.glb");
useGLTF.preload("/pieces/medieval/sun/k.glb");
