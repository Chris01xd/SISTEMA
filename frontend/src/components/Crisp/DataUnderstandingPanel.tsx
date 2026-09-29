import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { useTranslation } from "@/i18n/useTranslation";
import { extractErrorMessage } from "@/lib/errorMessage";
import { computeEda, createDataSource, listDataSources, listLoadingUnits } from "@/services/crispApi";
import type { DataSource } from "@/types/crisp";

const SOURCE_TYPES: DataSource["source_type"][] = [
  "IOT_SENSOR",
  "SCADA",
  "ERP",
  "MAINTENANCE_ORDERS",
  "FAILURE_HISTORY",
  "MANUFACTURER_DATA",
  "MANUAL_ENTRY",
];
const SENSOR_VARIABLES = ["TEMPERATURE", "VIBRATION", "HYDRAULIC_PRESSURE", "FUEL_CONSUMPTION", "ENGINE_RPM", "SPEED"];

const QUALITY_COLOR: Record<string, string> = {
  HIGH: "text-status-green",
  MEDIUM: "text-status-yellow",
  LOW: "text-status-red",
  UNASSESSED: "text-gray-500",
};

interface Props {
  projectId: string;
  canEdit: boolean;
}

export function DataUnderstandingPanel({ projectId, canEdit }: Props) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [name, setName] = useState("");
  const [sourceType, setSourceType] = useState(SOURCE_TYPES[0]);
  const [loadingUnitId, setLoadingUnitId] = useState("");
  const [variable, setVariable] = useState(SENSOR_VARIABLES[0]);
  const [error, setError] = useState<string | null>(null);

  const sources = useQuery({ queryKey: ["crisp-datasources", projectId], queryFn: () => listDataSources(projectId) });
  const units = useQuery({ queryKey: ["mining-loading-units"], queryFn: listLoadingUnits });

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ["crisp-datasources", projectId] });

  const create = useMutation({
    mutationFn: () =>
      createDataSource({
        project_id: projectId,
        name,
        source_type: sourceType,
        loading_unit_id: loadingUnitId || undefined,
        measured_variable: sourceType === "IOT_SENSOR" ? variable : "",
      }),
    onSuccess: () => {
      setName("");
      invalidate();
    },
    onError: (err) => setError(extractErrorMessage(err, t("crisp.common.error"))),
  });

  const eda = useMutation({
    mutationFn: (sourceId: string) => computeEda(sourceId),
    onSuccess: invalidate,
    onError: (err) => setError(extractErrorMessage(err, t("crisp.common.error"))),
  });

  const fieldClass = "rounded bg-gray-200 px-2 py-1.5 text-sm text-gray-900 dark:bg-gray-800 dark:text-white";

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{t("crisp.data.title")}</h3>

      {canEdit && (
        <div className="flex flex-wrap items-end gap-2 rounded-lg border border-gray-300 bg-gray-100 p-3 dark:border-gray-700 dark:bg-gray-900/60">
          <label className="flex flex-col text-xs text-gray-600 dark:text-gray-300">
            {t("crisp.data.name")}
            <input value={name} onChange={(e) => setName(e.target.value)} className={fieldClass} />
          </label>
          <label className="flex flex-col text-xs text-gray-600 dark:text-gray-300">
            {t("crisp.data.type")}
            <select value={sourceType} onChange={(e) => setSourceType(e.target.value as DataSource["source_type"])} className={fieldClass}>
              {SOURCE_TYPES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col text-xs text-gray-600 dark:text-gray-300">
            {t("crisp.projects.loadingUnits")}
            <select value={loadingUnitId} onChange={(e) => setLoadingUnitId(e.target.value)} className={fieldClass}>
              <option value="">{t("crisp.common.none")}</option>
              {units.data?.map((u) => (
                <option key={u.id} value={u.id}>
                  {u.code}
                </option>
              ))}
            </select>
          </label>
          {sourceType === "IOT_SENSOR" && (
            <label className="flex flex-col text-xs text-gray-600 dark:text-gray-300">
              {t("crisp.data.variable")}
              <select value={variable} onChange={(e) => setVariable(e.target.value)} className={fieldClass}>
                {SENSOR_VARIABLES.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
          )}
          <button
            onClick={() => create.mutate()}
            disabled={!name || create.isPending}
            className="rounded bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-50"
          >
            {t("crisp.data.addSource")}
          </button>
        </div>
      )}

      {error && (
        <p className="rounded-lg bg-red-100 p-2 text-xs text-red-700 ring-1 ring-red-300 dark:bg-red-950/40 dark:text-red-300 dark:ring-red-800/50">
          {error}
        </p>
      )}

      <div className="space-y-3">
        {sources.data?.map((source) => (
          <div key={source.id} className="rounded-lg border border-gray-300 bg-gray-100 p-3 dark:border-gray-700 dark:bg-gray-900/60">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <p className="text-sm font-semibold text-gray-900 dark:text-white">{source.name}</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {source.source_type} {source.measured_variable && `· ${source.measured_variable}`}
                </p>
              </div>
              <span className={`text-xs font-semibold ${QUALITY_COLOR[source.quality_level]}`}>
                {t("crisp.data.quality")}: {source.quality_level}
              </span>
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-gray-600 dark:text-gray-300 sm:grid-cols-4">
              <span>
                {t("crisp.data.records")}: {source.record_count}
              </span>
              <span>
                {t("crisp.data.missingPct")}: {source.missing_values_pct}%
              </span>
              <span>
                {t("crisp.data.duplicates")}: {source.duplicate_records}
              </span>
              <span>
                {t("crisp.data.outliers")}: {source.outlier_count}
              </span>
            </div>
            {source.eda_report?.descriptive_statistics && (
              <div className="mt-2 rounded bg-gray-200 p-2 text-xs text-gray-700 dark:bg-gray-800 dark:text-gray-300">
                <p className="mb-1 font-semibold">{t("crisp.data.edaSummary")}</p>
                <pre className="overflow-x-auto whitespace-pre-wrap">{JSON.stringify(source.eda_report.descriptive_statistics, null, 2)}</pre>
              </div>
            )}
            {canEdit && (
              <button
                onClick={() => eda.mutate(source.id)}
                disabled={eda.isPending}
                className="mt-2 rounded bg-gray-300 px-2 py-1 text-xs font-semibold text-gray-800 hover:bg-gray-400 dark:bg-gray-700 dark:text-gray-100"
              >
                {t("crisp.data.computeEda")}
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
