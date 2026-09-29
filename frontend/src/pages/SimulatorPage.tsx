import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { RunControlPanel } from "@/components/Simulator/RunControlPanel";
import { ScenarioForm } from "@/components/Simulator/ScenarioForm";
import { useTwinWebSocket } from "@/hooks/useTwinWebSocket";
import { useTranslation } from "@/i18n/useTranslation";
import { api } from "@/services/api";
import { useAuthStore } from "@/store/authStore";
import type { EmergencyEventCreate, Simulation } from "@/types";

interface SimulatorPageProps {
  hospitalId: string;
  onSimulationReady?: (simulationId: string) => void;
}

export function SimulatorPage({ hospitalId, onSimulationReady }: SimulatorPageProps) {
  useTwinWebSocket();
  const token = useAuthStore((s) => s.token);
  const { t } = useTranslation();
  const [eventId, setEventId] = useState<string | null>(null);
  const [simulation, setSimulation] = useState<Simulation | null>(null);

  function extractErrorMessage(err: unknown): string {
    const anyErr = err as { response?: { status?: number; data?: { detail?: string } } };
    if (anyErr?.response?.status === 403) return t("simulator.errorForbidden");
    if (anyErr?.response?.status === 401) return t("simulator.errorUnauthorized");
    const detail = anyErr?.response?.data?.detail;
    return typeof detail === "string" ? detail : t("simulator.errorGeneric");
  }

  const createEvent = useMutation({
    mutationFn: async (payload: EmergencyEventCreate) => {
      const res = await api.post(`/hospitals/${hospitalId}/emergency-events`, payload);
      return res.data;
    },
    onSuccess: (data) => {
      setEventId(data.id);
      setSimulation(null);
    },
  });

  const runSimulation = useMutation({
    mutationFn: async ({ horizon, strategy }: { horizon: number; strategy: string }) => {
      const res = await api.post<Simulation>("/simulations", {
        event_id: eventId,
        horizon_minutes: horizon,
        timestep_minutes: 1,
        use_ai_optimizer: strategy !== "greedy",
        assignment_strategy: strategy,
      });
      return res.data;
    },
    onSuccess: (data) => {
      setSimulation(data);
      onSimulationReady?.(data.id);
    },
  });

  // Polls the simulation's own status as a WS-independent fallback so the
  // user always sees COMPLETED/FAILED even if the WebSocket stream was
  // missed or hasn't connected — background execution and live streaming are
  // two separate concerns (see backend/app/services/simulation_service.py).
  const statusQuery = useQuery({
    queryKey: ["simulation-status", simulation?.id],
    queryFn: async () => {
      const res = await api.get<Simulation>(`/simulations/${simulation!.id}`);
      return res.data;
    },
    enabled: Boolean(simulation?.id),
    refetchInterval: (query) =>
      query.state.data && ["COMPLETED", "FAILED"].includes(query.state.data.status) ? false : 1000,
  });

  const liveSimulation = statusQuery.data ?? simulation;

  if (!hospitalId) {
    return (
      <p className="p-6 text-gray-500 dark:text-gray-400">{t("simulator.needHospital")}</p>
    );
  }

  return (
    <div className="grid gap-6 p-4 sm:p-6 lg:grid-cols-2">
      <ScenarioForm
        hospitalId={hospitalId}
        onSubmit={(payload) => createEvent.mutate(payload)}
        isSubmitting={createEvent.isPending}
      />
      <div className="space-y-4">
        {createEvent.isError && (
          <p className="rounded-lg bg-red-100 p-3 text-xs text-red-700 ring-1 ring-red-300 dark:bg-red-950/40 dark:text-red-300 dark:ring-red-800/50">
            {extractErrorMessage(createEvent.error)}
          </p>
        )}
        {eventId && (
          <p className="break-all rounded-lg bg-emerald-100 p-3 text-xs text-emerald-700 ring-1 ring-emerald-300 dark:bg-emerald-950/40 dark:text-emerald-300 dark:ring-emerald-800/50">
            {t("simulator.eventCreated")}: {eventId}
          </p>
        )}
        {!token && (
          <p className="rounded-lg bg-amber-100 p-3 text-xs text-amber-700 ring-1 ring-amber-300 dark:bg-amber-950/40 dark:text-amber-300 dark:ring-amber-800/50">
            {t("simulator.loginToRun")}
          </p>
        )}

        <RunControlPanel
          simulation={liveSimulation}
          isSubmitting={runSimulation.isPending}
          canRun={Boolean(eventId) && Boolean(token)}
          onRun={(horizon, strategy) => runSimulation.mutate({ horizon, strategy })}
        />

        {runSimulation.isError && (
          <p className="rounded-lg bg-red-100 p-3 text-xs text-red-700 ring-1 ring-red-300 dark:bg-red-950/40 dark:text-red-300 dark:ring-red-800/50">
            {extractErrorMessage(runSimulation.error)}
          </p>
        )}
      </div>
    </div>
  );
}
