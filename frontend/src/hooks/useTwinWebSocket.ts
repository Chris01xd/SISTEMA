import { useEffect } from "react";

import { connectTwinWebSocket } from "@/services/api";
import { useHospitalStore } from "@/store/hospitalStore";
import type { TwinWebSocketMessage } from "@/types";

export function useTwinWebSocket(): void {
  const applyTick = useHospitalStore((s) => s.applyTick);
  const addRecommendation = useHospitalStore((s) => s.addRecommendation);

  useEffect(() => {
    const ws = connectTwinWebSocket((data) => {
      const msg = data as TwinWebSocketMessage;
      if (msg.type === "simulation_tick") {
        applyTick(msg as never);
      } else if (msg.type === "ai_recommendation") {
        addRecommendation(msg as never);
      }
    });
    return () => ws.close();
  }, [applyTick, addRecommendation]);
}
