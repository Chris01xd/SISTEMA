import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { useTranslation } from "@/i18n/useTranslation";
import { extractErrorMessage } from "@/lib/errorMessage";
import {
  detectDataDrift,
  detectModelDrift,
  listDeployments,
  listDrift,
  listMonitoring,
  refreshMonitoring,
} from "@/services/crispApi";

interface Props {
  projectId: string;
}

export function MonitoringPanel({ projectId }: Props) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [deploymentId, setDeploymentId] = useState("");
  const [error, setError] = useState<string | null>(null);

  const deployments = useQuery({ queryKey: ["crisp-deployments", projectId], queryFn: () => listDeployments(projectId) });
  const monitoring = useQuery({
    queryKey: ["crisp-monitoring", deploymentId],
    queryFn: () => listMonitoring(deploymentId),
    enabled: !!deploymentId,
  });
  const drift = useQuery({ queryKey: ["crisp-drift", deploymentId], queryFn: () => listDrift(deploymentId), enabled: !!deploymentId });

  const refresh = useMutation({
    mutationFn: () => refreshMonitoring(deploymentId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["crisp-monitoring", deploymentId] }),
    onError: (err) => setError(extractErrorMessage(err, t("crisp.common.error"))),
  });
  const dataDrift = useMutation({
    mutationFn: () => detectDataDrift(deploymentId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["crisp-drift", deploymentId] }),
    onError: (err) => setError(extractErrorMessage(err, t("crisp.common.error"))),
  });
  const modelDrift = useMutation({
    mutationFn: () => detectModelDrift(deploymentId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["crisp-drift", deploymentId] }),
    onError: (err) => setError(extractErrorMessage(err, t("crisp.common.error"))),
  });

  const latest = monitoring.data?.[0];
  const chartData = (monitoring.data ?? [])
    .slice()
    .reverse()
    .map((m, i) => ({ i, predictions: m.prediction_count, alerts: m.alerts_generated_count }));

  const fieldClass = "rounded bg-gray-200 px-2 py-1.5 text-sm text-gray-900 dark:bg-gray-800 dark:text-white";

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{t("crisp.monitoring.title")}</h3>

      <select value={deploymentId} onChange={(e) => setDeploymentId(e.target.value)} className={fieldClass}>
        <option value="">{t("crisp.common.none")}</option>
        {deployments.data?.map((d) => (
          <option key={d.id} value={d.id}>
            {d.version}
          </option>
        ))}
      </select>

      {deploymentId && (
        <>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => refresh.mutate()}
              disabled={refresh.isPending}
              className="rounded bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-50"
            >
              {t("crisp.monitoring.refresh")}
            </button>
            <button
              onClick={() => dataDrift.mutate()}
              disabled={dataDrift.isPending}
              className="rounded bg-gray-300 px-3 py-1.5 text-xs font-semibold text-gray-800 hover:bg-gray-400 dark:bg-gray-700 dark:text-gray-100"
            >
              {t("crisp.monitoring.checkDataDrift")}
            </button>
            <button
              onClick={() => modelDrift.mutate()}
              disabled={modelDrift.isPending}
              className="rounded bg-gray-300 px-3 py-1.5 text-xs font-semibold text-gray-800 hover:bg-gray-400 dark:bg-gray-700 dark:text-gray-100"
            >
              {t("crisp.monitoring.checkModelDrift")}
            </button>
          </div>

          {error && (
            <p className="rounded-lg bg-red-100 p-2 text-xs text-red-700 ring-1 ring-red-300 dark:bg-red-950/40 dark:text-red-300 dark:ring-red-800/50">
              {error}
            </p>
          )}

          {latest && (
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              <Stat label={t("crisp.monitoring.accuracy")} value={latest.current_accuracy?.toFixed(2) ?? "—"} />
              <Stat label={t("crisp.monitoring.predictions")} value={latest.prediction_count} />
              <Stat label={t("crisp.monitoring.alertsGenerated")} value={latest.alerts_generated_count} />
              <Stat
                label={t("crisp.monitoring.needsRetraining")}
                value={latest.needs_retraining ? "⚠" : "✓"}
                accent={latest.needs_retraining ? "text-status-red" : "text-status-green"}
              />
            </div>
          )}

          {chartData.length > 1 && (
            <div className="rounded-xl border border-gray-300 bg-gray-100 p-3 dark:border-gray-700 dark:bg-gray-900/60">
              <ResponsiveContainer width="100%" height={160}>
                <LineChart data={chartData}>
                  <XAxis dataKey="i" stroke="#6b7280" />
                  <YAxis stroke="#6b7280" />
                  <Tooltip />
                  <Line type="monotone" dataKey="predictions" stroke="#6366f1" dot={false} />
                  <Line type="monotone" dataKey="alerts" stroke="#dc2626" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          <div>
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-200">{t("crisp.monitoring.driftHistory")}</h4>
            <div className="mt-1 space-y-1">
              {drift.data?.map((d) => (
                <p key={d.id} className="text-xs text-gray-600 dark:text-gray-300">
                  <span className="font-semibold">{d.drift_type}</span> · {d.variable_name ?? "—"} · score {d.drift_score.toFixed(4)} ·{" "}
                  <span className={d.is_significant ? "text-status-red" : "text-status-green"}>
                    {d.is_significant ? t("crisp.monitoring.significant") : t("crisp.monitoring.notSignificant")}
                  </span>
                </p>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: string | number; accent?: string }) {
  return (
    <div className="rounded-xl border border-gray-300 bg-gray-100 p-3 dark:border-gray-700 dark:bg-gray-900/60">
      <p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">{label}</p>
      <p className={`mt-1 text-xl font-bold ${accent ?? "text-gray-900 dark:text-white"}`}>{value}</p>
    </div>
  );
}
