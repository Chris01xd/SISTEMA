import { useState } from "react";

import { useTranslation } from "@/i18n/useTranslation";
import type { Simulation } from "@/types";

interface RunControlPanelProps {
  onRun: (horizonMinutes: number, strategy: string) => void;
  simulation: Simulation | null;
  isSubmitting?: boolean;
  canRun: boolean;
}

const STATUS_CLASS: Record<Simulation["status"], string> = {
  PENDING: "bg-gray-200 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
  RUNNING: "bg-amber-100 text-amber-700 ring-1 ring-amber-300 dark:bg-amber-950/50 dark:text-amber-300 dark:ring-amber-800/50",
  COMPLETED:
    "bg-emerald-100 text-emerald-700 ring-1 ring-emerald-300 dark:bg-emerald-950/50 dark:text-emerald-300 dark:ring-emerald-800/50",
  FAILED: "bg-red-100 text-red-700 ring-1 ring-red-300 dark:bg-red-950/50 dark:text-red-300 dark:ring-red-800/50",
};

export function RunControlPanel({ onRun, simulation, isSubmitting, canRun }: RunControlPanelProps) {
  const [horizon, setHorizon] = useState(360);
  const [strategy, setStrategy] = useState("greedy");
  const { t } = useTranslation();

  const HORIZON_OPTIONS = [
    { label: t("runControl.horizon30"), value: 30 },
    { label: t("runControl.horizon60"), value: 60 },
    { label: t("runControl.horizon360"), value: 360 },
    { label: t("runControl.horizon1440"), value: 1440 },
  ];

  const STATUS_LABEL: Record<Simulation["status"], string> = {
    PENDING: t("runControl.statusPending"),
    RUNNING: t("runControl.statusRunning"),
    COMPLETED: t("runControl.statusCompleted"),
    FAILED: t("runControl.statusFailed"),
  };

  return (
    <div className="space-y-4 rounded-xl border border-gray-300 bg-gray-100 p-4 dark:border-gray-700 dark:bg-gray-900/60 sm:p-5">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-gray-600 dark:text-gray-300">
        {t("runControl.title")}
      </h3>

      <div className="flex flex-wrap gap-2">
        {HORIZON_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            onClick={() => setHorizon(opt.value)}
            className={`rounded-full px-3 py-1 text-xs ${
              horizon === opt.value
                ? "bg-red-600 text-white"
                : "bg-gray-200 text-gray-700 hover:bg-gray-300 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700"
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>

      <label className="flex flex-col text-sm text-gray-600 dark:text-gray-300">
        {t("runControl.strategyLabel")}
        <select
          value={strategy}
          onChange={(e) => setStrategy(e.target.value)}
          className="mt-1 w-full rounded bg-gray-200 px-2 py-1.5 text-gray-900 dark:bg-gray-800 dark:text-white"
        >
          <option value="greedy">{t("runControl.strategyGreedy")}</option>
          <option value="genetic">{t("runControl.strategyGenetic")}</option>
          <option value="rl">{t("runControl.strategyRl")}</option>
        </select>
      </label>

      <button
        onClick={() => onRun(horizon, strategy)}
        disabled={isSubmitting || !canRun}
        className="w-full rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {isSubmitting ? t("runControl.starting") : t("runControl.start")}
      </button>
      {!canRun && <p className="text-xs text-gray-500">{t("runControl.needEvent")}</p>}

      {simulation && (
        <div className="space-y-2 rounded-lg bg-gray-200/60 p-3 text-xs text-gray-700 dark:bg-gray-800/60 dark:text-gray-300">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="break-all">
              {t("runControl.idLabel")}: {simulation.id}
            </span>
            <span className={`rounded-full px-2 py-0.5 font-semibold ${STATUS_CLASS[simulation.status]}`}>
              {STATUS_LABEL[simulation.status]}
            </span>
          </div>
          <p>
            {t("runControl.horizonLabel")}: {simulation.horizon_minutes} {t("runControl.minutes")} | Δt:{" "}
            {simulation.timestep_minutes} {t("runControl.minutes")}
          </p>
          {simulation.status === "RUNNING" && (
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-gray-300 dark:bg-gray-700">
              <div className="h-full w-1/3 animate-pulse rounded-full bg-amber-500" />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
