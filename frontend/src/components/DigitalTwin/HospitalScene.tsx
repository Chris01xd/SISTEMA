import { ContactShadows, Grid, Html, OrbitControls, RoundedBox } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useMemo, useState } from "react";

import { useTranslation } from "@/i18n/useTranslation";
import type { HospitalTwinSnapshot } from "@/types";

import { Ambulance } from "./Ambulance";
import { CityBackdrop } from "./CityBackdrop";
import { FloorSlab, SLAB_THICKNESS } from "./FloorSlab";
import { RoomBlock } from "./RoomBlock";

interface HospitalSceneProps {
  snapshot: HospitalTwinSnapshot;
}

const FLOOR_HEIGHT = 5;
const ROOM_SPACING = 6.5; // room footprint (4) plus interior walkway room
const WALL_HEIGHT = FLOOR_HEIGHT - 1;
const FOOTPRINT_PADDING = 3.5;

export function HospitalScene({ snapshot }: HospitalSceneProps) {
  const [selected, setSelected] = useState<{ id: string; kind: "room" | "bed" } | null>(null);
  const { t } = useTranslation();

  // The backend seeds every floor's rooms across one shared, flat floor-plan
  // sketch (e.g. the ER sits at z=-6, the ICU at z=0, Wards at z=12 — all on
  // the SAME xz plane), so stacking them by floor index while keeping those
  // raw x/z produced disconnected islands instead of a building: each floor
  // "inherited" a different slice of that flat sketch. Laying each floor's
  // own rooms out in a small grid, centered at the origin, makes every level
  // occupy the same footprint and actually stack into a tower.
  const floorsWithRooms = useMemo(() => {
    return snapshot.floors
      .slice()
      .sort((a, b) => a.level - b.level)
      .map((floor, floorIndex) => {
        const rooms = snapshot.rooms.filter((r) => r.floor_id === floor.id);
        const cols = Math.max(1, Math.ceil(Math.sqrt(rooms.length)));
        const rows = Math.max(1, Math.ceil(rooms.length / cols));
        const positionedRooms = rooms.map((room, i) => ({
          room: {
            ...room,
            position_x: ((i % cols) - (cols - 1) / 2) * ROOM_SPACING,
            position_y: floorIndex * FLOOR_HEIGHT,
            position_z: (Math.floor(i / cols) - (rows - 1) / 2) * ROOM_SPACING,
          },
          beds: snapshot.beds.filter((b) => b.room_id === room.id),
        }));
        return {
          floor,
          floorIndex,
          rooms: positionedRooms,
          footprintWidth: cols * ROOM_SPACING,
          footprintDepth: rows * ROOM_SPACING,
        };
      });
  }, [snapshot]);

  const selectedEntity = useMemo(() => {
    if (!selected) return null;
    if (selected.kind === "room") return snapshot.rooms.find((r) => r.id === selected.id);
    return snapshot.beds.find((b) => b.id === selected.id);
  }, [selected, snapshot]);

  const {
    buildingWidth,
    buildingDepth,
    cameraPosition,
    floorLabelX,
    floorLabelZ,
    roofY,
    helipadX,
    helipadZ,
    helipadRadius,
  } = useMemo(() => {
    const width = Math.max(ROOM_SPACING, ...floorsWithRooms.map((f) => f.footprintWidth)) + FOOTPRINT_PADDING;
    const depth = Math.max(ROOM_SPACING, ...floorsWithRooms.map((f) => f.footprintDepth)) + FOOTPRINT_PADDING;
    const floorCount = floorsWithRooms.length;
    const span = Math.max(width, depth, 12) + 6;
    const dist = span * 1.15 + floorCount * FLOOR_HEIGHT * 0.55;
    return {
      buildingWidth: width,
      buildingDepth: depth,
      cameraPosition: [dist * 0.65, dist * 0.55 + floorCount * FLOOR_HEIGHT * 0.3, dist * 0.65] as [
        number,
        number,
        number,
      ],
      floorLabelX: -width / 2 - 1.4,
      floorLabelZ: -depth / 2 - 1.4,
      roofY: (floorCount - 1) * FLOOR_HEIGHT + WALL_HEIGHT + SLAB_THICKNESS,
      // Tucked in a corner of the roof, clear of the cross, the sign post,
      // and the top floor's rooms.
      helipadX: width * 0.27,
      helipadZ: depth * 0.27,
      helipadRadius: Math.min(width, depth) * 0.22,
    };
  }, [floorsWithRooms]);

  return (
    <div className="relative h-full w-full bg-gradient-to-b from-slate-200 to-slate-300 dark:from-gray-950 dark:to-gray-900">
      <Canvas shadows dpr={[1, 2]} camera={{ position: cameraPosition, fov: 45 }}>
        <hemisphereLight args={["#e0f2fe", "#1e293b", 0.65]} />
        <ambientLight intensity={0.3} />
        <directionalLight
          position={[18, 26, 14]}
          intensity={1.2}
          castShadow
          shadow-mapSize={[2048, 2048]}
          shadow-camera-left={-35}
          shadow-camera-right={35}
          shadow-camera-top={35}
          shadow-camera-bottom={-35}
        />
        <fog attach="fog" args={["#94a3b8", 50, 120]} />

        <Grid
          position={[0, -0.03, 0]}
          args={[10, 10]}
          cellSize={2}
          cellThickness={0.5}
          cellColor="#94a3b8"
          sectionSize={10}
          sectionThickness={1.1}
          sectionColor="#475569"
          fadeDistance={80}
          fadeStrength={1.2}
          infiniteGrid
        />

        {/* The clear zone must be a circle big enough to clear the building's
            CORNERS, not just its side midpoints — half the diagonal, not
            half the width/depth — otherwise houses end up tucked right
            against the tower's corners even though they're outside the
            "radius" on each flat side. */}
        <CityBackdrop
          clearRadius={Math.sqrt(buildingWidth ** 2 + buildingDepth ** 2) / 2 + 5}
        />

        {/* Ground plinth the tower sits on, distinguishing the building's
            footprint from the open site grid. */}
        <mesh position={[0, -0.18, 0]} receiveShadow>
          <boxGeometry args={[buildingWidth + 1.5, 0.36, buildingDepth + 1.5]} />
          <meshStandardMaterial color="#94a3b8" roughness={0.9} />
        </mesh>

        {/* Entrance canopy and ambulance bay at the ground floor's front. */}
        <group position={[0, 0, buildingDepth / 2]}>
          <RoundedBox args={[buildingWidth * 0.5, 0.15, 3]} radius={0.04} smoothness={2} position={[0, 2.3, 1.7]} castShadow>
            <meshStandardMaterial color="#94a3b8" roughness={0.8} />
          </RoundedBox>
          {[-1, 1].map((side) => (
            <mesh key={side} position={[side * buildingWidth * 0.22, 1.15, 3.1]} castShadow>
              <cylinderGeometry args={[0.08, 0.08, 2.3, 8]} />
              <meshStandardMaterial color="#94a3b8" />
            </mesh>
          ))}
          <Ambulance position={[-1.5, 0, 2.6]} rotationY={0.12} />
          <Ambulance position={[1.4, 0, 3.3]} rotationY={-0.2} />
        </group>

        {floorsWithRooms.map(({ floor, floorIndex }) => (
          <FloorSlab
            key={floor.id}
            y={floorIndex * FLOOR_HEIGHT}
            width={buildingWidth}
            depth={buildingDepth}
            wallHeight={WALL_HEIGHT}
            isRoof={floorIndex === floorsWithRooms.length - 1}
          />
        ))}

        {floorsWithRooms.map(({ floor, floorIndex }) => (
          <Html key={floor.id} position={[floorLabelX, floorIndex * FLOOR_HEIGHT + 0.4, floorLabelZ]} zIndexRange={[10, 0]}>
            <div className="pointer-events-none whitespace-nowrap rounded-md bg-white/90 px-2.5 py-1 text-xs font-semibold uppercase tracking-wide text-gray-700 shadow ring-1 ring-gray-300 dark:bg-gray-900/90 dark:text-gray-200 dark:ring-gray-700">
              {floor.name}
            </div>
          </Html>
        ))}

        {floorsWithRooms.flatMap(({ rooms }) => rooms).map(({ room, beds }) => (
          <RoomBlock
            key={room.id}
            room={room}
            beds={beds}
            onSelect={(id, kind) => setSelected({ id, kind })}
          />
        ))}

        {/* Rooftop red cross marking, like the ones painted on real hospital
            helipads/roofs, so the building reads as a hospital at a glance. */}
        <group position={[0, roofY + 0.02, 0]}>
          <mesh>
            <boxGeometry args={[Math.min(buildingWidth, buildingDepth) * 0.4, 0.02, Math.min(buildingWidth, buildingDepth) * 0.14]} />
            <meshStandardMaterial color="#dc2626" />
          </mesh>
          <mesh>
            <boxGeometry args={[Math.min(buildingWidth, buildingDepth) * 0.14, 0.02, Math.min(buildingWidth, buildingDepth) * 0.4]} />
            <meshStandardMaterial color="#dc2626" />
          </mesh>
        </group>

        {/* Rooftop helipad: a yellow ring around a dark pad with an "H"
            built from plain boxes (no font dependency), tucked in a corner
            clear of the cross, the sign post, and the top floor's rooms. */}
        <group position={[helipadX, roofY + 0.02, helipadZ]}>
          <mesh rotation={[-Math.PI / 2, 0, 0]}>
            <ringGeometry args={[helipadRadius * 0.86, helipadRadius, 48]} />
            <meshStandardMaterial color="#facc15" />
          </mesh>
          <mesh position={[0, -0.005, 0]} rotation={[-Math.PI / 2, 0, 0]}>
            <circleGeometry args={[helipadRadius * 0.86, 48]} />
            <meshStandardMaterial color="#1f2937" />
          </mesh>
          <mesh position={[-helipadRadius * 0.32, 0.01, 0]}>
            <boxGeometry args={[helipadRadius * 0.16, 0.02, helipadRadius * 0.9]} />
            <meshStandardMaterial color="#f8fafc" />
          </mesh>
          <mesh position={[helipadRadius * 0.32, 0.01, 0]}>
            <boxGeometry args={[helipadRadius * 0.16, 0.02, helipadRadius * 0.9]} />
            <meshStandardMaterial color="#f8fafc" />
          </mesh>
          <mesh position={[0, 0.01, 0]}>
            <boxGeometry args={[helipadRadius * 0.8, 0.02, helipadRadius * 0.16]} />
            <meshStandardMaterial color="#f8fafc" />
          </mesh>
        </group>

        {/* Rooftop sign post naming the building as a hospital. */}
        <mesh position={[0, roofY + 1.1, -buildingDepth / 4]}>
          <cylinderGeometry args={[0.06, 0.06, 2.2, 8]} />
          <meshStandardMaterial color="#94a3b8" />
        </mesh>
        <Html position={[0, roofY + 2.3, -buildingDepth / 4]} center zIndexRange={[10, 0]}>
          <div className="pointer-events-none flex items-center gap-1.5 whitespace-nowrap rounded-md bg-white px-3 py-1.5 text-sm font-bold uppercase tracking-wide text-red-600 shadow-lg ring-2 ring-red-600 dark:bg-gray-900 dark:text-red-400 dark:ring-red-500">
            <span aria-hidden>✚</span>
            <span>
              {t("twin.hospitalSign")}
              {snapshot.hospital.name ? ` — ${snapshot.hospital.name}` : ""}
            </span>
          </div>
        </Html>

        <ContactShadows position={[0, -0.36, 0]} opacity={0.4} scale={buildingWidth + buildingDepth} blur={2.4} far={20} />

        <OrbitControls
          enableDamping
          dampingFactor={0.08}
          minDistance={8}
          maxDistance={100}
          maxPolarAngle={Math.PI / 2.05}
          makeDefault
        />
      </Canvas>

      {selectedEntity && (
        <div className="absolute right-2 top-2 max-h-[70%] w-[calc(100vw-1rem)] max-w-72 overflow-auto rounded-lg bg-white/95 p-4 text-sm shadow-xl ring-1 ring-gray-300 dark:bg-gray-900/95 dark:ring-white/10 sm:right-4 sm:top-4 sm:w-72">
          <h3 className="mb-2 font-semibold text-gray-900 dark:text-white">{t("twin.entityInspector")}</h3>
          <pre className="whitespace-pre-wrap text-xs text-gray-600 dark:text-gray-300">
            {JSON.stringify(selectedEntity, null, 2)}
          </pre>
        </div>
      )}

      <div className="absolute left-2 top-2 flex max-w-[calc(100vw-1rem)] flex-wrap gap-x-3 gap-y-1 rounded-lg bg-white/90 px-3 py-2 text-xs shadow ring-1 ring-gray-300 dark:bg-gray-900/80 dark:ring-gray-700 sm:left-4 sm:top-4">
        <LegendDot color="#16a34a" label={t("twin.legendAvailable")} />
        <LegendDot color="#eab308" label={t("twin.legendHighDemand")} />
        <LegendDot color="#dc2626" label={t("twin.legendCritical")} />
      </div>
    </div>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
      <span className="text-gray-700 dark:text-gray-200">{label}</span>
    </div>
  );
}
