import { useMemo } from "react";

interface CityBackdropProps {
  /** Nothing is placed closer than this to the origin, so the city stays
   * clear of the hospital's own footprint. */
  clearRadius: number;
}

interface CityBuilding {
  x: number;
  z: number;
  width: number;
  depth: number;
  height: number;
  isHouse: boolean;
  color: string;
}

interface CityTree {
  x: number;
  z: number;
  scale: number;
}

// Muted, low-contrast tones — the hospital's blue glass and status colors
// stay the only saturated things in the scene, so the city reads as distant
// backdrop rather than competing for attention. Warm tones for houses, cool
// grays for taller blocks, so the two archetypes read as different at a glance.
const HOUSE_COLORS = ["#e7d9c4", "#ddcdb2", "#e3d3ba", "#d8c8ad"];
const BLOCK_COLORS = ["#cbd5e1", "#b6c2d1", "#a8b4c2", "#c3c9d1"];
const ROOF_COLORS = ["#a15c3f", "#8f5238", "#9c5f42"];

// Small deterministic PRNG (mulberry32) so the city's layout is stable across
// re-renders instead of reshuffling every time the twin snapshot updates.
function mulberry32(seed: number) {
  let state = seed;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const MIN_GAP = 1.4;
// Half-width of the road slabs below (their depth/width arg is 3), plus a
// buffer, so nothing gets placed with a wheel on the asphalt.
const ROAD_CLEARANCE = 1.5 + 0.5;

function overlapsRoad(x: number, z: number, halfSize: number): boolean {
  return Math.abs(z) < ROAD_CLEARANCE + halfSize || Math.abs(x) < ROAD_CLEARANCE + halfSize;
}

function placeBuildings(rand: () => number, clearRadius: number, target: number): CityBuilding[] {
  const placed: CityBuilding[] = [];
  let attempts = 0;
  while (placed.length < target && attempts < target * 40) {
    attempts++;
    const angle = rand() * Math.PI * 2;
    const radius = clearRadius + rand() * clearRadius * 1.6;
    const x = Math.cos(angle) * radius;
    const z = Math.sin(angle) * radius;
    const width = 1.3 + rand() * 1.7;
    const depth = 1.3 + rand() * 1.7;
    const footprintRadius = Math.max(width, depth) / 2;

    if (overlapsRoad(x, z, footprintRadius)) continue;

    const collides = placed.some((b) => {
      const dx = b.x - x;
      const dz = b.z - z;
      const otherRadius = Math.max(b.width, b.depth) / 2;
      return Math.hypot(dx, dz) < footprintRadius + otherRadius + MIN_GAP;
    });
    if (collides) continue;

    const isHouse = rand() < 0.55;
    placed.push({
      x,
      z,
      width,
      depth,
      height: isHouse ? 0.9 + rand() * 0.5 : 1.8 + rand() * 3.2,
      isHouse,
      color: isHouse
        ? HOUSE_COLORS[Math.floor(rand() * HOUSE_COLORS.length)]
        : BLOCK_COLORS[Math.floor(rand() * BLOCK_COLORS.length)],
    });
  }
  return placed;
}

/** A deliberately low-detail ring of background buildings, a couple of
 * streets, and a few trees around the hospital plot — enough so the twin
 * doesn't read as floating alone on an empty grid, spaced apart and shaped
 * (pitched-roof houses, flat-roof blocks with window bands) so they don't
 * read as bare, crowded cubes, without competing with the hospital for
 * visual attention. */
export function CityBackdrop({ clearRadius }: CityBackdropProps) {
  const { buildings, trees } = useMemo(() => {
    const rand = mulberry32(1337);
    const buildingList = placeBuildings(rand, clearRadius, 22);
    const treeList: CityTree[] = [];
    let treeAttempts = 0;
    while (treeList.length < 16 && treeAttempts < 16 * 40) {
      treeAttempts++;
      const angle = rand() * Math.PI * 2;
      const radius = clearRadius * 0.85 + rand() * clearRadius * 1.4;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      if (overlapsRoad(x, z, 0.3)) continue;
      treeList.push({ x, z, scale: 0.7 + rand() * 0.6 });
    }
    return { buildings: buildingList, trees: treeList };
  }, [clearRadius]);

  const armLength = clearRadius * 2.4;
  const roadStart = clearRadius + 0.6;

  return (
    <group>
      {/* Streets crossing near the hospital, gapped so they don't run
          straight through its plot — reads like the hospital sits on a
          corner lot at the intersection. */}
      {[1, -1].map((sign) => (
        <mesh key={`road-x-${sign}`} position={[sign * (roadStart + armLength / 2), -0.02, 0]} receiveShadow>
          <boxGeometry args={[armLength, 0.02, 3]} />
          <meshStandardMaterial color="#9ca3af" roughness={1} />
        </mesh>
      ))}
      {[1, -1].map((sign) => (
        <mesh key={`road-z-${sign}`} position={[0, -0.02, sign * (roadStart + armLength / 2)]} receiveShadow>
          <boxGeometry args={[3, 0.02, armLength]} />
          <meshStandardMaterial color="#9ca3af" roughness={1} />
        </mesh>
      ))}

      {buildings.map((b, i) =>
        b.isHouse ? (
          <group key={i} position={[b.x, 0, b.z]}>
            <mesh position={[0, b.height / 2, 0]} receiveShadow castShadow>
              <boxGeometry args={[b.width, b.height, b.depth]} />
              <meshStandardMaterial color={b.color} roughness={0.9} />
            </mesh>
            <mesh
              position={[0, b.height + (b.width + b.depth) * 0.14, 0]}
              rotation={[0, Math.PI / 4, 0]}
              scale={[1, 1, b.depth / b.width]}
              castShadow
            >
              <coneGeometry args={[b.width * 0.78, (b.width + b.depth) * 0.28, 4]} />
              <meshStandardMaterial color={ROOF_COLORS[i % ROOF_COLORS.length]} roughness={0.85} />
            </mesh>
          </group>
        ) : (
          <group key={i} position={[b.x, 0, b.z]}>
            <mesh position={[0, b.height / 2, 0]} receiveShadow castShadow>
              <boxGeometry args={[b.width, b.height, b.depth]} />
              <meshStandardMaterial color={b.color} roughness={0.92} />
            </mesh>
            <mesh position={[0, b.height * 0.36, 0]}>
              <boxGeometry args={[b.width + 0.01, Math.min(0.16, b.height * 0.08), b.depth + 0.01]} />
              <meshStandardMaterial color="#64748b" roughness={0.4} />
            </mesh>
            <mesh position={[0, b.height * 0.68, 0]}>
              <boxGeometry args={[b.width + 0.01, Math.min(0.16, b.height * 0.08), b.depth + 0.01]} />
              <meshStandardMaterial color="#64748b" roughness={0.4} />
            </mesh>
            <mesh position={[0, b.height + 0.06, 0]} receiveShadow castShadow>
              <boxGeometry args={[b.width * 0.92, 0.12, b.depth * 0.92]} />
              <meshStandardMaterial color="#eef2f6" roughness={0.7} />
            </mesh>
          </group>
        )
      )}

      {trees.map((tr, i) => (
        <group key={i} position={[tr.x, 0, tr.z]} scale={tr.scale}>
          <mesh position={[0, 0.3, 0]}>
            <cylinderGeometry args={[0.06, 0.08, 0.6, 6]} />
            <meshStandardMaterial color="#8a6a4a" roughness={1} />
          </mesh>
          <mesh position={[0, 0.85, 0]}>
            <coneGeometry args={[0.45, 1, 7]} />
            <meshStandardMaterial color="#5f8a5a" roughness={0.9} />
          </mesh>
        </group>
      ))}
    </group>
  );
}
