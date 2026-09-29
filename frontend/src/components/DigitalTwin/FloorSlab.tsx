import { RoundedBox } from "@react-three/drei";
import { DoubleSide } from "three";

interface FloorSlabProps {
  y: number;
  width: number;
  depth: number;
  wallHeight: number;
  isRoof: boolean;
}

export const SLAB_THICKNESS = 0.3;
const GLASS_PROPS = {
  color: "#93c5fd",
  transparent: true,
  opacity: 0.1,
  roughness: 0.15,
  metalness: 0.1,
  side: DoubleSide,
} as const;

/** One story of the building "maqueta": a solid floor slab plus four glass
 * perimeter walls, so rooms read as being inside a real hospital floor
 * instead of floating disconnected in space. Optionally caps with a roof. */
export function FloorSlab({ y, width, depth, wallHeight, isRoof }: FloorSlabProps) {
  return (
    <group position={[0, y, 0]}>
      <RoundedBox
        args={[width, SLAB_THICKNESS, depth]}
        radius={0.06}
        smoothness={2}
        position={[0, -SLAB_THICKNESS / 2, 0]}
        receiveShadow
        castShadow
      >
        <meshStandardMaterial color="#cbd5e1" roughness={0.9} />
      </RoundedBox>

      <mesh position={[0, wallHeight / 2, -depth / 2]} receiveShadow>
        <boxGeometry args={[width, wallHeight, 0.06]} />
        <meshStandardMaterial {...GLASS_PROPS} />
      </mesh>
      <mesh position={[0, wallHeight / 2, depth / 2]} receiveShadow>
        <boxGeometry args={[width, wallHeight, 0.06]} />
        <meshStandardMaterial {...GLASS_PROPS} />
      </mesh>
      <mesh position={[-width / 2, wallHeight / 2, 0]} receiveShadow>
        <boxGeometry args={[0.06, wallHeight, depth]} />
        <meshStandardMaterial {...GLASS_PROPS} />
      </mesh>
      <mesh position={[width / 2, wallHeight / 2, 0]} receiveShadow>
        <boxGeometry args={[0.06, wallHeight, depth]} />
        <meshStandardMaterial {...GLASS_PROPS} />
      </mesh>

      {isRoof && (
        <RoundedBox
          args={[width, SLAB_THICKNESS, depth]}
          radius={0.06}
          smoothness={2}
          position={[0, wallHeight + SLAB_THICKNESS / 2, 0]}
          receiveShadow
          castShadow
        >
          <meshStandardMaterial color="#e2e8f0" roughness={0.85} />
        </RoundedBox>
      )}
    </group>
  );
}
