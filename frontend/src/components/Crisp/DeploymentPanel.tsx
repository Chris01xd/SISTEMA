import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { useTranslation } from "@/i18n/useTranslation";
import { extractErrorMessage } from "@/lib/errorMessage";
import { createPrediction, deployModel, listDeployments, listExperiments, listLoadingUnits, listModelsForExperiment } from "@/services/crispApi";
import type { ModelPrediction } from "@/types/crisp";

const STATE_COLOR: Record<string, string> = {
  NORMAL: "text-status-green",
  WARNING: "text-status-yellow",
  CRITICAL: "text-status-red",
  OUT_OF_SERVICE: "text-gray-500",
};

interface Props {
  projectId: string;
  canEdit: boolean;
}

export function DeploymentPanel({ projectId, canEdit }: Props) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [modelId, setModelId] = useState("");
  const [unitIds, setUnitIds] = useState<string[]>([]);
  const [warning, setWarning] = useState(0.4);
  const [critical, setCritical] = useState(0.75);
  const [predictUnitId, setPredictUnitId] = useState("");
  const [lastPrediction, setLastPrediction] = useState<ModelPrediction | null>(null);
  const [error, setError] = useState<string | null>(null);

  const experiments = useQuery({ queryKey: ["crisp-experiments", projectId], queryFn: () => listExperiments(projectId) });
  const completed = experiments.data?.filter((e) => e.status === "COMPLETED") ?? [];
  const modelLists = useQuery({
    queryKey: ["crisp-deploy-models", projectId, completed.map((e) => e.id).join(",")],
    queryFn: async () => (await Promise.all(completed.map((e) => listModelsForExperiment(e.id)))).flat(),
    enabled: completed.length > 0,
  });
  const units = useQuery({ queryKey: ["mining-loading-units"], queryFn: listLoadingUnits });
  const deployments = useQuery({ queryKey: ["crisp-deployments", projectId], queryFn: () => listDeployments(projectId) });

  const deploy = useMutation({
    mutationFn: () =>
      deployModel(modelId, { project_id: projectId, loading_unit_ids: unitIds, alert_thresholds: { warning, critical } }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["crisp-deployments", projectId] }),
    onError: (err) => setError(extractErrorMessage(err, t("crisp.common.error"))),
  });

  const predict = useMutation({
    mutationFn: (deploymentId: string) => createPrediction(deploymentId, predictUnitId),
    onSuccess: setLastPrediction,
    onError: (err) => setError(extractErrorMessage(err, t("crisp.common.error"))),
  });

  const fieldClass = "rounded bg-gray-200 px-2 py-1.5 text-sm text-gray-900 dark:bg-gray-800 dark:text-white";

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{t("crisp.deployment.title")}</h3>

      {canEdit && (
        <div className="grid grid-cols-1 gap-2 rounded-lg border border-gray-300 bg-gray-100 p-3 dark:border-gray-700 dark:bg-gray-900/60 sm:grid-cols-2 lg:grid-cols-4">
          <label className="flex flex-col text-xs text-gray-600 dark:text-gray-300">
            {t("crisp.modeling.models")}
            <select value={modelId} onChange={(e) => setModelId(e.target.value)} className={fieldClass}>
              <option value="">{t("crisp.common.none")}</option>
              {modelLists.data?.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} {m.version}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col text-xs text-gray-600 dark:text-gray-300">
            {t("crisp.projects.loadingUnits")}
            <select
              multiple
              value={unitIds}
              onChange={(e) => setUnitIds(Array.from(e.target.selectedOptions, (o) => o.value))}
              className={fieldClass}
            >
              {units.data?.map((u) => (
                <option key={u.id} value={u.id}>
                  {u.code}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col text-xs text-gray-600 dark:text-gray-300">
            {t("crisp.deployment.thresholds")} (warning)
            <input type="number" step="0.05" value={warning} onChange={(e) => setWarning(Number(e.target.value))} className={fieldClass} />
          </label>
          <label className="flex flex-col text-xs text-gray-600 dark:text-gray-300">
            {t("crisp.deployment.thresholds")} (critical)
            <input type="number" step="0.05" value={critical} onChange={(e) => setCritical(Number(e.target.value))} className={fieldClass} />
          </label>
          <button
            onClick={() => deploy.mutate()}
            disabled={!modelId || unitIds.length === 0 || deploy.isPending}
            className="rounded bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-50 sm:col-span-2 lg:col-span-4"
          >
            {t("crisp.deployment.deploy")}
          </button>
        </div>
      )}

      {error && (
        <p className="rounded-lg bg-red-100 p-2 text-xs text-red-700 ring-1 ring-red-300 dark:bg-red-950/40 dark:text-red-300 dark:ring-red-800/50">
          {error}
        </p>
      )}

      <div className="space-y-3">
        {deployments.data?.map((d) => (
          <div key={d.id} className="rounded-lg border border-gray-300 bg-gray-100 p-3 dark:border-gray-700 dark:bg-gray-900/60">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-sm font-semibold text-gray-900 dark:text-white">{d.version}</span>
              <span className="text-xs font-semibold text-status-green">{d.service_status}</span>
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400">{d.api_endpoint}</p>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              {t("crisp.deployment.nextRetraining")}: {d.next_retraining_date ?? "—"}
            </p>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <select value={predictUnitId} onChange={(e) => setPredictUnitId(e.target.value)} className={fieldClass}>
                <option value="">{t("crisp.common.none")}</option>
                {units.data?.map((u) => (
                  <option key={u.id} value={u.id}>
                    {u.code}
                  </option>
                ))}
              </select>
              <button
                onClick={() => predict.mutate(d.id)}
                disabled={!predictUnitId || predict.isPending}
                className="rounded bg-gray-300 px-2 py-1 text-xs font-semibold text-gray-800 hover:bg-gray-400 dark:bg-gray-700 dark:text-gray-100"
              >
                {t("crisp.deployment.predict")}
              </button>
            </div>
            {lastPrediction && lastPrediction.deployment_id === d.id && (
              <div className="mt-2 rounded bg-gray-200 p-2 text-xs dark:bg-gray-800">
                <p className="font-semibold text-gray-700 dark:text-gray-200">{t("crisp.deployment.predictionResult")}</p>
                <p className={STATE_COLOR[lastPrediction.equipment_state]}>
                  {lastPrediction.equipment_state} — risk {lastPrediction.risk_level.toFixed(2)}
                </p>
                <p className="text-gray-600 dark:text-gray-300">{lastPrediction.maintenance_recommendation}</p>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
