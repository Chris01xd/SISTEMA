import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { useTranslation } from "@/i18n/useTranslation";
import { extractErrorMessage } from "@/lib/errorMessage";
import { evaluateModel, listEvaluations, listExperiments, listModelsForExperiment } from "@/services/crispApi";
import type { EvaluationDecision } from "@/types/crisp";

const DECISIONS: EvaluationDecision[] = ["APPROVED", "REJECTED", "RETURN_TO_DATA_UNDERSTANDING", "RETURN_TO_DATA_PREPARATION"];

interface Props {
  projectId: string;
  canEdit: boolean;
}

export function EvaluationPanel({ projectId, canEdit }: Props) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [selectedModelId, setSelectedModelId] = useState("");
  const [decision, setDecision] = useState<EvaluationDecision>("APPROVED");
  const [opinion, setOpinion] = useState("");
  const [justification, setJustification] = useState("");
  const [anticipationHours, setAnticipationHours] = useState<number | "">("");
  const [mtbfImpact, setMtbfImpact] = useState<number | "">("");
  const [error, setError] = useState<string | null>(null);

  const experiments = useQuery({ queryKey: ["crisp-experiments", projectId], queryFn: () => listExperiments(projectId) });
  const completedExperiments = experiments.data?.filter((e) => e.status === "COMPLETED") ?? [];

  const modelLists = useQuery({
    queryKey: ["crisp-eval-models", projectId, completedExperiments.map((e) => e.id).join(",")],
    queryFn: async () => {
      const lists = await Promise.all(completedExperiments.map((e) => listModelsForExperiment(e.id)));
      return lists.flat();
    },
    enabled: completedExperiments.length > 0,
  });

  const evaluations = useQuery({
    queryKey: ["crisp-evaluations", selectedModelId],
    queryFn: () => listEvaluations(selectedModelId),
    enabled: !!selectedModelId,
  });

  const submit = useMutation({
    mutationFn: () =>
      evaluateModel(selectedModelId, {
        decision,
        specialist_opinion: opinion,
        justification,
        operational_metrics: { anticipation_time_hours: anticipationHours === "" ? null : anticipationHours },
        business_metrics_estimate: { estimated_mtbf_impact: mtbfImpact === "" ? null : mtbfImpact },
      }),
    onSuccess: () => {
      setOpinion("");
      setJustification("");
      queryClient.invalidateQueries({ queryKey: ["crisp-evaluations", selectedModelId] });
    },
    onError: (err) => setError(extractErrorMessage(err, t("crisp.common.error"))),
  });

  const fieldClass = "rounded bg-gray-200 px-2 py-1.5 text-sm text-gray-900 dark:bg-gray-800 dark:text-white";

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{t("crisp.evaluation.title")}</h3>

      <label className="flex flex-col text-xs text-gray-600 dark:text-gray-300">
        {t("crisp.modeling.models")}
        <select value={selectedModelId} onChange={(e) => setSelectedModelId(e.target.value)} className={fieldClass}>
          <option value="">{t("crisp.common.none")}</option>
          {modelLists.data?.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name} {m.version}
            </option>
          ))}
        </select>
      </label>

      {selectedModelId && canEdit && (
        <div className="space-y-2 rounded-lg border border-gray-300 bg-gray-100 p-3 dark:border-gray-700 dark:bg-gray-900/60">
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <label className="flex flex-col text-xs text-gray-600 dark:text-gray-300">
              {t("crisp.evaluation.anticipationHours")}
              <input
                type="number"
                value={anticipationHours}
                onChange={(e) => setAnticipationHours(e.target.value === "" ? "" : Number(e.target.value))}
                className={fieldClass}
              />
            </label>
            <label className="flex flex-col text-xs text-gray-600 dark:text-gray-300">
              {t("crisp.evaluation.mtbfImpact")}
              <input
                type="number"
                value={mtbfImpact}
                onChange={(e) => setMtbfImpact(e.target.value === "" ? "" : Number(e.target.value))}
                className={fieldClass}
              />
            </label>
          </div>
          <p className="text-xs italic text-gray-500 dark:text-gray-400">{t("crisp.evaluation.businessNote")}</p>
          <label className="flex flex-col text-xs text-gray-600 dark:text-gray-300">
            {t("crisp.evaluation.opinion")}
            <textarea value={opinion} onChange={(e) => setOpinion(e.target.value)} rows={2} className={fieldClass} />
          </label>
          <label className="flex flex-col text-xs text-gray-600 dark:text-gray-300">
            {t("crisp.evaluation.justification")}
            <textarea value={justification} onChange={(e) => setJustification(e.target.value)} rows={2} className={fieldClass} />
          </label>
          <label className="flex flex-col text-xs text-gray-600 dark:text-gray-300">
            {t("crisp.evaluation.decision")}
            <select value={decision} onChange={(e) => setDecision(e.target.value as EvaluationDecision)} className={fieldClass}>
              {DECISIONS.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </label>
          <button
            onClick={() => submit.mutate()}
            disabled={!justification || submit.isPending}
            className="rounded bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-50"
          >
            {t("crisp.evaluation.submit")}
          </button>
        </div>
      )}

      {error && (
        <p className="rounded-lg bg-red-100 p-2 text-xs text-red-700 ring-1 ring-red-300 dark:bg-red-950/40 dark:text-red-300 dark:ring-red-800/50">
          {error}
        </p>
      )}

      {selectedModelId && (
        <div>
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-200">{t("crisp.evaluation.history")}</h4>
          <div className="mt-1 space-y-2">
            {evaluations.data?.map((ev) => (
              <div key={ev.id} className="rounded bg-gray-100 p-2 text-xs text-gray-700 dark:bg-gray-900/60 dark:text-gray-300">
                <span className="font-semibold">{ev.decision}</span> — {ev.justification}
              </div>
            ))}
            {evaluations.data?.length === 0 && <p className="text-xs text-gray-500 dark:text-gray-400">{t("crisp.audit.empty")}</p>}
          </div>
        </div>
      )}
    </div>
  );
}
