import { Html, RoundedBox } from "@react-three/drei";
import { useMemo, useState } from "react";
import * as THREE from "three";

import { useTranslation } from "@/i18n/useTranslation";
import type { BedTwin, RoomTwin } from "@/types";

import { bedStatusToColor, STATUS_HEX } from "./statusColors";

interface RoomBlockProps {
  room: RoomTwin;
  beds: BedTwin[];
  onSelect: (id: string, kind: "room" | "bed") => void;
}

const ROOM_SIZE = 4;
// A plain box's edges (not RoundedBox's, whose beveled corners add stray
// diagonal facet lines) give a clean rectangular outline per room.
const ROOM_EDGES_GEOMETRY = new THREE.EdgesGeometry(new THREE.BoxGeometry(ROOM_SIZE, 0.16, ROOM_SIZE));

const WALL_HEIGHT = 1.8;
const WALL_THICKNESS = 0.08;
const DOOR_WIDTH = 1.1;
const HALF_WALL = (ROOM_SIZE - DOOR_WIDTH) / 2;
const WALL_MATERIAL_PROPS = { color: "#e5e7eb", transparent: true, opacity: 0.65, roughness: 0.9 } as const;

/** Renders one hospital room as a status-tinted platform with a glowing
 * outline (matching the twin's traffic-light status), a floating code/type
 * label, and its beds as small mattress+pillow shapes arranged on a grid
 * inside the footprint, colored by occupancy. */
export function RoomBlock({ room, beds, onSelect }: RoomBlockProps) {
  const [hovered, setHovered] = useState(false);
  const { t } = useTranslation();
  const statusColor = STATUS_HEX[room.status] ?? "#6b7280";
  const occupiedBeds = beds.filter((b) => b.status === "OCCUPIED").length;

  const bedPositions = useMemo(() => {
    const cols = Math.max(1, Math.ceil(Math.sqrt(beds.length)));
    const spacing = ROOM_SIZE / (cols + 1);
    return beds.map((bed, i) => {
      const row = Math.floor(i / cols);
      const col = i % cols;
      return { bed, x: (col + 1) * spacing - ROOM_SIZE / 2, z: (row + 1) * spacing - ROOM_SIZE / 2 };
    });
  }, [beds]);

  return (
    <group position={[room.position_x, room.position_y, room.position_z]}>
      <RoundedBox
        args={[ROOM_SIZE, 0.16, ROOM_SIZE]}
        radius={0.08}
        smoothness={2}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
        }}
        onPointerOut={() => setHovered(false)}
        onClick={(e) => {
          e.stopPropagation();
          onSelect(room.id, "room");
        }}
        receiveShadow
      >
        <meshStandardMaterial color={statusColor} transparent opacity={hovered ? 0.5 : 0.26} roughness={0.65} />
      </RoundedBox>
      <lineSegments geometry={ROOM_EDGES_GEOMETRY} position={[0, 0.002, 0]}>
        <lineBasicMaterial color={statusColor} />
      </lineSegments>

      {/* Partition walls, so rooms read as enclosed spaces rather than open
          platforms. The south wall (+Z, facing the corridor) has a gap for
          the doorway. */}
      <mesh position={[0, WALL_HEIGHT / 2, -ROOM_SIZE / 2]}>
        <boxGeometry args={[ROOM_SIZE, WALL_HEIGHT, WALL_THICKNESS]} />
        <meshStandardMaterial {...WALL_MATERIAL_PROPS} />
      </mesh>
      <mesh position={[-ROOM_SIZE / 2, WALL_HEIGHT / 2, 0]}>
        <boxGeometry args={[WALL_THICKNESS, WALL_HEIGHT, ROOM_SIZE]} />
        <meshStandardMaterial {...WALL_MATERIAL_PROPS} />
      </mesh>
      <mesh position={[ROOM_SIZE / 2, WALL_HEIGHT / 2, 0]}>
        <boxGeometry args={[WALL_THICKNESS, WALL_HEIGHT, ROOM_SIZE]} />
        <meshStandardMaterial {...WALL_MATERIAL_PROPS} />
      </mesh>
      <mesh position={[-(DOOR_WIDTH / 2 + HALF_WALL / 2), WALL_HEIGHT / 2, ROOM_SIZE / 2]}>
        <boxGeometry args={[HALF_WALL, WALL_HEIGHT, WALL_THICKNESS]} />
        <meshStandardMaterial {...WALL_MATERIAL_PROPS} />
      </mesh>
      <mesh position={[DOOR_WIDTH / 2 + HALF_WALL / 2, WALL_HEIGHT / 2, ROOM_SIZE / 2]}>
        <boxGeometry args={[HALF_WALL, WALL_HEIGHT, WALL_THICKNESS]} />
        <meshStandardMaterial {...WALL_MATERIAL_PROPS} />
      </mesh>
      {/* Door leaf, swung open into the room from the hinge at the gap's edge */}
      <group position={[-DOOR_WIDTH / 2, 0, ROOM_SIZE / 2]} rotation={[0, -0.75, 0]}>
        <mesh position={[DOOR_WIDTH / 2, WALL_HEIGHT * 0.42, 0]} castShadow>
          <boxGeometry args={[DOOR_WIDTH * 0.94, WALL_HEIGHT * 0.82, 0.05]} />
          <meshStandardMaterial color="#a16207" roughness={0.55} />
        </mesh>
      </group>

      {bedPositions.map(({ bed, x, z }) => {
        const bedColor = bedStatusToColor(bed.status);
        return (
          <group
            key={bed.id}
            position={[x, 0.28, z]}
            onClick={(e) => {
              e.stopPropagation();
              onSelect(bed.id, "bed");
            }}
          >
            <RoundedBox args={[0.55, 0.22, 0.95]} radius={0.05} smoothness={2} castShadow receiveShadow>
              <meshStandardMaterial color={bedColor} roughness={0.5} />
            </RoundedBox>
            <RoundedBox args={[0.48, 0.12, 0.22]} radius={0.04} smoothness={2} position={[0, 0.16, -0.32]} castShadow>
              <meshStandardMaterial color="#f8fafc" roughness={0.9} />
            </RoundedBox>
          </group>
        );
      })}

      <Html position={[0, hovered ? WALL_HEIGHT + 0.7 : WALL_HEIGHT + 0.3, 0]} center distanceFactor={11} zIndexRange={[10, 0]}>
        <div
          className={`pointer-events-none select-none whitespace-nowrap rounded-md bg-white/90 px-2 py-1 text-center shadow ring-1 ring-gray-300 transition-all dark:bg-gray-900/90 dark:ring-gray-700 ${
            hovered ? "text-xs" : "text-[10px] opacity-80"
          }`}
        >
          <div className="font-semibold text-gray-800 dark:text-gray-100">{room.code}</div>
          {hovered && (
            <div className="text-gray-500 dark:text-gray-400">
              {room.room_type} · {occupiedBeds}/{beds.length} {t("twin.beds")}
            </div>
          )}
        </div>
      </Html>
    </group>
  );
}
