import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { useTranslation } from "@/i18n/useTranslation";
import { useHospitalStore } from "@/store/hospitalStore";

import { AlertsPanel } from "./AlertsPanel";
import { KpiCard } from "./KpiCard";

const STATUS_ACCENT: Record<string, "green" | "yellow" | "red"> = {
  GREEN: "green",
  YELLOW: "yellow",
  RED: "red",
};

export function ExecutiveDashboard() {
  const latestTick = useHospitalStore((s) => s.latestTick);
  const tickHistory = useHospitalStore((s) => s.tickHistory);
  const recommendations = useHospitalStore((s) => s.recommendations);
  const globalStatus = useHospitalStore((s) => s.globalStatus);
  const { t } = useTranslation();

  const census = latestTick?.census;
  const occupancyPct = census ? Math.round((census.beds_occupied / Math.max(census.beds_total, 1)) * 100) : 0;
  const chartData = tickHistory.map((tick) => ({ tick: tick.tick, saturation: tick.saturation_index }));

  return (
    <div className="space-y-4 p-4 sm:space-y-6 sm:p-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">{t("dashboard.title")}</h2>
        <span
          className={`rounded-full px-3 py-1 text-xs font-semibold ${
            globalStatus === "GREEN"
              ? "bg-status-green/20 text-status-green"
              : globalStatus === "YELLOW"
              ? "bg-status-yellow/20 text-status-yellow"
              : "bg-status-red/20 text-status-red"
          }`}
        >
          {t("dashboard.globalStatus")}: {globalStatus}
        </span>
      </div>

      {!latestTick && (
        <p className="rounded-lg bg-gray-100 p-3 text-sm text-gray-600 ring-1 ring-gray-300 dark:bg-gray-900/60 dark:text-gray-400 dark:ring-gray-800">
          {t("dashboard.noDataPrefix")} <strong>{t("dashboard.noDataLink")}</strong> {t("dashboard.noDataSuffix")}
        </p>
      )}

      <div className="grid grid-cols-2 gap-3 sm:gap-4 md:grid-cols-4">
        <KpiCard label={t("dashboard.bedOccupancy")} value={`${occupancyPct}%`} accent={STATUS_ACCENT[globalStatus]} />
        <KpiCard label={t("dashboard.patientsWaiting")} value={census?.patients_waiting ?? 0} />
        <KpiCard
          label={t("dashboard.saturationIndex")}
          value={latestTick ? latestTick.saturation_index.toFixed(2) : "0.00"}
          accent={STATUS_ACCENT[globalStatus]}
        />
        <KpiCard label={t("dashboard.activeStaff")} value={`${census?.staff_active ?? 0}/${census?.staff_total ?? 0}`} />
      </div>

      <div className="rounded-xl border border-gray-300 bg-gray-100 p-3 dark:border-gray-700 dark:bg-gray-900/60 sm:p-4">
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-gray-600 dark:text-gray-300">
          {t("dashboard.realtimeSaturation")}
        </h3>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={chartData}>
            <XAxis dataKey="tick" stroke="#6b7280" />
            <YAxis domain={[0, 1]} stroke="#6b7280" />
            <Tooltip />
            <Area type="monotone" dataKey="saturation" stroke="#dc2626" fill="#dc262633" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <AlertsPanel recommendations={recommendations} />
    </div>
  );
}
