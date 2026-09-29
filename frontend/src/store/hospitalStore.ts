import { create } from "zustand";

import type { AIRecommendationMessage, HospitalTwinSnapshot, SimulationTickMessage, StatusColor } from "@/types";

const MAX_TICK_HISTORY = 200;

interface HospitalState {
  snapshot: HospitalTwinSnapshot | null;
  latestTick: SimulationTickMessage | null;
  tickHistory: SimulationTickMessage[];
  globalStatus: StatusColor;
  recommendations: AIRecommendationMessage[];
  setSnapshot: (snapshot: HospitalTwinSnapshot) => void;
  applyTick: (tick: SimulationTickMessage) => void;
  addRecommendation: (rec: AIRecommendationMessage) => void;
  resetTickHistory: () => void;
}

export const useHospitalStore = create<HospitalState>((set) => ({
  snapshot: null,
  latestTick: null,
  tickHistory: [],
  globalStatus: "GREEN",
  recommendations: [],
  setSnapshot: (snapshot) => set({ snapshot }),
  applyTick: (tick) =>
    set((state) => {
      // A new simulation restarting at tick 0/1 means past history is stale.
      const isNewRun = state.latestTick !== null && tick.tick < state.latestTick.tick;
      const nextHistory = isNewRun ? [tick] : [...state.tickHistory, tick].slice(-MAX_TICK_HISTORY);
      return { latestTick: tick, globalStatus: tick.status_band, tickHistory: nextHistory };
    }),
  addRecommendation: (rec) =>
    set((state) => ({ recommendations: [rec, ...state.recommendations].slice(0, 20) })),
  resetTickHistory: () => set({ tickHistory: [], latestTick: null }),
}));
