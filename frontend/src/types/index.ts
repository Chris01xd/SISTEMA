export type StatusColor = "GREEN" | "YELLOW" | "RED";

export interface Hospital {
  id: string;
  name: string;
  location: string;
  total_capacity: number;
}

export interface RoomTwin {
  id: string;
  floor_id: string;
  code: string;
  room_type: string;
  capacity: number;
  position_x: number;
  position_y: number;
  position_z: number;
  status: StatusColor;
}

export interface BedTwin {
  id: string;
  room_id: string;
  code: string;
  bed_type: string;
  status: "AVAILABLE" | "OCCUPIED" | "CLEANING" | "OUT_OF_SERVICE";
  patient_id: string | null;
}

export interface FloorTwin {
  id: string;
  hospital_id: string;
  level: number;
  name: string;
  floor_type: string;
}

export interface HospitalTwinSnapshot {
  hospital: Hospital;
  floors: FloorTwin[];
  rooms: RoomTwin[];
  beds: BedTwin[];
}

export interface SimulationCensus {
  patients_waiting: number;
  patients_in_treatment: number;
  patients_icu: number;
  patients_ward: number;
  patients_deceased: number;
  patients_discharged: number;
  beds_total: number;
  beds_occupied: number;
  staff_total: number;
  staff_active: number;
  total_patients_seen: number;
}

export interface SimulationTickMessage {
  type: "simulation_tick";
  simulation_id: string;
  tick: number;
  census: SimulationCensus;
  saturation_index: number;
  status_band: StatusColor;
  staff_fatigue: number;
}

export interface AIRecommendationMessage {
  type: "ai_recommendation";
  simulation_id: string;
  message: string;
}

export type TwinWebSocketMessage = SimulationTickMessage | AIRecommendationMessage | { type: string; [k: string]: unknown };

export interface EmergencyEventCreate {
  hospital_id: string;
  name: string;
  event_type: "EARTHQUAKE" | "EXPLOSION" | "TRANSPORT" | "INDUSTRIAL" | "OTHER";
  magnitude: number;
  expected_victims: number;
  severity_distribution: {
    minor: number;
    delayed: number;
    immediate: number;
    critical: number;
    expectant: number;
  };
  arrival_curve: "surge_decay" | "linear" | "poisson_steady";
  arrival_window_minutes: number;
}

export interface Simulation {
  id: string;
  event_id: string;
  horizon_minutes: number;
  timestep_minutes: number;
  status: "PENDING" | "RUNNING" | "COMPLETED" | "FAILED";
  started_at: string | null;
  finished_at: string | null;
}
