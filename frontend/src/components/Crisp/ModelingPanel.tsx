import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { useTranslation } from "@/i18n/useTranslation";
import { extractErrorMessage } from "@/lib/errorMessage";
import {
  compareExperiments,
  createExperiment,
  listExperiments,
  listModelsForExperiment,
  publishModel,
  trainExperiment,
} from "@/services/crispApi";
import type { Algorithm, ModelObjective } from "@/types/crisp";

const OBJECTIVES: ModelObjective[] = [
  "FAILURE_CLASSIFICATION",
  "ANOMALY_DETECTION",
  "FAILURE_PREDICTION",
  "REMAINING_USEFUL_LIFE",
  "OPERATIONAL_VARIABLE_FORECAST",
];
const ALGORITHMS: Algorithm[] = ["RANDOM_FOREST", "XGBOOST", "LSTM", "ISOLATION_FOREST", "REGRESSION"];

const STATUS_COLOR: Record<string, string> = {
  COMPLETED: "text-status-green",
  TRAINING: "text-status-yellow",
  FAILED: "text-status-red",
  DRAFT: "text-gray-500",
  QUEUED: "text-gray-500",
};

interface Props {
  projectId: string;
  datasetVersionId: string;
  canEdit: boolean;
}

/** `datasetVersionId` comes pre-filled from the most recent VALID version
 * created in phase 3 (see CrispProjectDetailPage), which the scientist can
 * override by pasting a different version id. */
export function ModelingPanel({ projectId, datasetVersionId, canEdit }: Props) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [versionId, setVersionId] = useState(datasetVersionId);
  const [objective, setObjective] = useState<ModelObjective>("FAILURE_CLASSIFICATION");
  const [algorithm, setAlgorithm] = useState<Algorithm>("RANDOM_FOREST");
  const [inputVars, setInputVars] = useState("");
  const [target, setTarget] = useState("failure_within_horizon");
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  const experiments = useQuery({ queryKey: ["crisp-experiments", projectId], queryFn: () => listExperiments(projectId) });
  const comparison = useQuery({ queryKey: ["crisp-compare", projectId], queryFn: () => compareExperiments(projectId) });

  const create = useMutation({
    mutationFn: () =>
      createExperiment({
        project_id: projectId,
        dataset_version_id: versionId,
        objective,
        algorithm,
        input_variables: inputVars.split(",").map((s) => s.trim()).filter(Boolean),
        target_variable: target,
        hyperparameters: {},
        validation_method: "TEMPORAL_HOLDOUT",
      }),
    onSuccess: () => {
      setInputVars("");
      queryClient.invalidateQueries({ queryKey: ["crisp-experiments", projectId] });
    },
    onError: (err) => setError(extractErrorMessage(err, t("crisp.common.error"))),
  });

  const train = useMutation({
    mutationFn: (id: string) => trainExperiment(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["crisp-experiments", projectId] });
      queryClient.invalidateQueries({ queryKey: ["crisp-compare", projectId] });
    },
    onError: (err) => setError(extractErrorMessage(err, t("crisp.common.error"))),
  });

  const publish = useMutation({
    mutationFn: (modelId: string) => publishModel(modelId),
    onSuccess: (_, modelId) => queryClient.invalidateQueries({ queryKey: ["crisp-models", modelId] }),
    onError: (err) => setError(extractErrorMessage(err, t("crisp.common.error"))),
  });

  const fieldClass = "rounded bg-gray-200 px-2 py-1.5 text-sm text-gray-900 dark:bg-gray-800 dark:text-white";

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{t("crisp.modeling.title")}</h3>

      {canEdit && (
        <div className="grid grid-cols-1 gap-2 rounded-lg border border-gray-300 bg-gray-100 p-3 dark:border-gray-700 dark:bg-gray-900/60 sm:grid-cols-2 lg:grid-cols-3">
          <label className="flex flex-col text-xs text-gray-600 dark:text-gray-300">
            {t("crisp.modeling.datasetVersion")}
            <input value={versionId} onChange={(e) => setVersionId(e.target.value)} className={fieldClass} placeholder="dataset_version_id" />
          </label>
          <label className="flex flex-col text-xs text-gray-600 dark:text-gray-300">
            {t("crisp.modeling.objective")}
            <select value={objective} onChange={(e) => setObjective(e.target.value as ModelObjective)} className={fieldClass}>
              {OBJECTIVES.map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col text-xs text-gray-600 dark:text-gray-300">
            {t("crisp.modeling.algorithm")}
            <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value as Algorithm)} className={fieldClass}>
              {ALGORITHMS.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col text-xs text-gray-600 dark:text-gray-300 sm:col-span-2">
            {t("crisp.modeling.inputVariables")}
            <input value={inputVars} onChange={(e) => setInputVars(e.target.value)} className={fieldClass} placeholder="temperature_xxx, vibration_xxx" />
          </label>
          <label className="flex flex-col text-xs text-gray-600 dark:text-gray-300">
            {t("crisp.modeling.targetVariable")}
            <input value={target} onChange={(e) => setTarget(e.target.value)} className={fieldClass} />
          </label>
          <button
            onClick={() => create.mutate()}
            disabled={!versionId || !inputVars || create.isPending}
            className="rounded bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-50 sm:col-span-2 lg:col-span-1"
          >
            {t("crisp.modeling.addExperiment")}
          </button>
        </div>
      )}

      {error && (
        <p className="rounded-lg bg-red-100 p-2 text-xs text-red-700 ring-1 ring-red-300 dark:bg-red-950/40 dark:text-red-300 dark:ring-red-800/50">
          {error}
        </p>
      )}

      <div className="space-y-3">
        {experiments.data?.map((exp) => {
          const compareRow = comparison.data?.find((c) => c.experiment.id === exp.id);
          return (
            <div key={exp.id} className="rounded-lg border border-gray-300 bg-gray-100 p-3 dark:border-gray-700 dark:bg-gray-900/60">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <p className="text-sm font-semibold text-gray-900 dark:text-white">
                    {exp.algorithm} — {exp.objective}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">{t("crisp.modeling.targetVariable")}: {exp.target_variable}</p>
                </div>
                <span className={`text-xs font-semibold ${STATUS_COLOR[exp.status]}`}>{exp.status}</span>
              </div>
              {compareRow && compareRow.metrics.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-3 text-xs text-gray-600 dark:text-gray-300">
                  {compareRow.metrics.map((m) => (
                    <span key={m.metric_name}>
                      {m.metric_name}: <strong>{m.metric_value.toFixed(3)}</strong>
                    </span>
                  ))}
                </div>
              )}
              <div className="mt-2 flex flex-wrap gap-2">
                {canEdit && exp.status !== "COMPLETED" && exp.status !== "TRAINING" && (
                  <button
                    onClick={() => train.mutate(exp.id)}
                    disabled={train.isPending}
                    className="rounded bg-gray-300 px-2 py-1 text-xs font-semibold text-gray-800 hover:bg-gray-400 dark:bg-gray-700 dark:text-gray-100"
                  >
                    {train.isPending && train.variables === exp.id ? t("crisp.modeling.training") : t("crisp.modeling.train")}
                  </button>
                )}
                {exp.status === "COMPLETED" && (
                  <button onClick={() => setExpanded(expanded === exp.id ? null : exp.id)} className="text-xs text-indigo-600 dark:text-indigo-400">
                    {t("crisp.modeling.models")}
                  </button>
                )}
              </div>
              {expanded === exp.id && <ExperimentModels experimentId={exp.id} canEdit={canEdit} onPublish={(id) => publish.mutate(id)} />}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ExperimentModels({ experimentId, canEdit, onPublish }: { experimentId: string; canEdit: boolean; onPublish: (id: string) => void }) {
  const { t } = useTranslation();
  const models = useQuery({ queryKey: ["crisp-models", experimentId], queryFn: () => listModelsForExperiment(experimentId) });

  return (
    <div className="mt-2 space-y-1 border-t border-gray-300 pt-2 text-xs dark:border-gray-700">
      {models.data?.map((m) => (
        <div key={m.id} className="flex items-center justify-between gap-2">
          <span className="text-gray-700 dark:text-gray-300">
            {m.name} {m.version} {m.is_published && `(${t("crisp.modeling.published")})`}
          </span>
          {canEdit && !m.is_published && (
            <button onClick={() => onPublish(m.id)} className="rounded bg-indigo-600 px-2 py-0.5 text-white">
              {t("crisp.modeling.publish")}
            </button>
          )}
        </div>
      ))}
    </div>
  );
}
