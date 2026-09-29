import { RoundedBox } from "@react-three/drei";

interface AmbulanceProps {
  position: [number, number, number];
  rotationY?: number;
}

const WHEEL_POSITIONS: Array<[number, number]> = [
  [-0.5, 0.75],
  [0.5, 0.75],
  [-0.5, -0.85],
  [0.5, -0.85],
];

/** A small stylized ambulance (boxes only, no model asset) parked at the
 * hospital entrance — a red cross, warning light bar, and side stripe are
 * enough to read as "ambulance" at this scale. */
export function Ambulance({ position, rotationY = 0 }: AmbulanceProps) {
  return (
    <group position={position} rotation={[0, rotationY, 0]}>
      <RoundedBox args={[1.1, 0.55, 2.4]} radius={0.06} smoothness={2} position={[0, 0.45, 0]} castShadow receiveShadow>
        <meshStandardMaterial color="#f8fafc" roughness={0.4} />
      </RoundedBox>
      <RoundedBox args={[1.0, 0.42, 0.75]} radius={0.05} smoothness={2} position={[0, 0.72, 0.95]} castShadow>
        <meshStandardMaterial color="#93c5fd" roughness={0.2} transparent opacity={0.6} />
      </RoundedBox>

      <mesh position={[0.56, 0.45, 0]}>
        <boxGeometry args={[0.02, 0.14, 2.3]} />
        <meshStandardMaterial color="#dc2626" />
      </mesh>
      <mesh position={[-0.56, 0.45, 0]}>
        <boxGeometry args={[0.02, 0.14, 2.3]} />
        <meshStandardMaterial color="#dc2626" />
      </mesh>

      <group position={[0, 0.46, -0.3]}>
        <mesh position={[0.57, 0, 0]}>
          <boxGeometry args={[0.02, 0.34, 0.1]} />
          <meshStandardMaterial color="#dc2626" />
        </mesh>
        <mesh position={[0.57, 0, 0]}>
          <boxGeometry args={[0.02, 0.1, 0.34]} />
          <meshStandardMaterial color="#dc2626" />
        </mesh>
      </group>

      <mesh position={[0, 0.75, 0.55]}>
        <boxGeometry args={[0.6, 0.08, 0.16]} />
        <meshStandardMaterial color="#1e293b" />
      </mesh>
      <mesh position={[-0.16, 0.79, 0.55]}>
        <boxGeometry args={[0.22, 0.03, 0.1]} />
        <meshStandardMaterial color="#dc2626" emissive="#dc2626" emissiveIntensity={0.9} />
      </mesh>
      <mesh position={[0.16, 0.79, 0.55]}>
        <boxGeometry args={[0.22, 0.03, 0.1]} />
        <meshStandardMaterial color="#3b82f6" emissive="#3b82f6" emissiveIntensity={0.9} />
      </mesh>

      {WHEEL_POSITIONS.map(([x, z], i) => (
        <mesh key={i} position={[x, 0.2, z]} rotation={[0, 0, Math.PI / 2]} castShadow>
          <cylinderGeometry args={[0.2, 0.2, 0.18, 12]} />
          <meshStandardMaterial color="#1f2937" roughness={0.8} />
        </mesh>
      ))}
    </group>
  );
}
