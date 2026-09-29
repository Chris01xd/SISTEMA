import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { useTranslation } from "@/i18n/useTranslation";
import { extractErrorMessage } from "@/lib/errorMessage";
import { updatePhaseData } from "@/services/crispApi";
import type { BusinessUnderstandingData, CrispPhase, KpiBaselineTarget } from "@/types/crisp";

const EMPTY: BusinessUnderstandingData = {
  business_problem: "",
  affected_loading_unit_ids: [],
  maintenance_processes_involved: [],
  general_objective: "",
  specific_objectives: [],
  interested_users: [],
  constraints: [],
  initial_risks: [],
  available_resources: [],
  kpis: [],
  expected_goals: "",
  business_success_criteria: [],
};

const toLines = (v: string[] | undefined) => (v ?? []).join("\n");
const fromLines = (v: string) => v.split("\n").map((s) => s.trim()).filter(Boolean);

interface Props {
  projectId: string;
  phase: CrispPhase;
  canEdit: boolean;
}

export function BusinessUnderstandingPanel({ projectId, phase, canEdit }: Props) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const data = { ...EMPTY, ...(phase.phase_data as Partial<BusinessUnderstandingData>) };

  const [problem, setProblem] = useState(data.business_problem);
  const [objective, setObjective] = useState(data.general_objective);
  const [specificObjectives, setSpecificObjectives] = useState(toLines(data.specific_objectives));
  const [interestedUsers, setInterestedUsers] = useState(toLines(data.interested_users));
  const [constraints, setConstraints] = useState(toLines(data.constraints));
  const [risks, setRisks] = useState(toLines(data.initial_risks));
  const [resources, setResources] = useState(toLines(data.available_resources));
  const [successCriteria, setSuccessCriteria] = useState(toLines(data.business_success_criteria));
  const [expectedGoals, setExpectedGoals] = useState(data.expected_goals);
  const [kpis, setKpis] = useState<KpiBaselineTarget[]>(data.kpis.length ? data.kpis : []);

  const save = useMutation({
    mutationFn: () =>
      updatePhaseData(phase.id, {
        business_problem: problem,
        affected_loading_unit_ids: data.affected_loading_unit_ids,
        maintenance_processes_involved: data.maintenance_processes_involved,
        general_objective: objective,
        specific_objectives: fromLines(specificObjectives),
        interested_users: fromLines(interestedUsers),
        constraints: fromLines(constraints),
        initial_risks: fromLines(risks),
        available_resources: fromLines(resources),
        kpis,
        expected_goals: expectedGoals,
        business_success_criteria: fromLines(successCriteria),
      }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["crisp-phases", projectId] }),
  });

  const updateKpi = (index: number, patch: Partial<KpiBaselineTarget>) =>
    setKpis((prev) => prev.map((k, i) => (i === index ? { ...k, ...patch } : k)));

  const fieldClass = "mt-1 w-full rounded bg-gray-200 px-2 py-1.5 text-sm text-gray-900 dark:bg-gray-800 dark:text-white disabled:opacity-60";
  const labelClass = "flex flex-col text-sm text-gray-600 dark:text-gray-300";

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{t("crisp.business.title")}</h3>

      <label className={labelClass}>
        {t("crisp.business.problem")}
        <textarea disabled={!canEdit} value={problem} onChange={(e) => setProblem(e.target.value)} rows={2} className={fieldClass} />
      </label>

      <label className={labelClass}>
        {t("crisp.business.generalObjective")}
        <textarea disabled={!canEdit} value={objective} onChange={(e) => setObjective(e.target.value)} rows={2} className={fieldClass} />
      </label>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <label className={labelClass}>
          {t("crisp.business.specificObjectives")}
          <textarea disabled={!canEdit} value={specificObjectives} onChange={(e) => setSpecificObjectives(e.target.value)} rows={3} className={fieldClass} />
        </label>
        <label className={labelClass}>
          {t("crisp.business.interestedUsers")}
          <textarea disabled={!canEdit} value={interestedUsers} onChange={(e) => setInterestedUsers(e.target.value)} rows={3} className={fieldClass} />
        </label>
        <label className={labelClass}>
          {t("crisp.business.constraints")}
          <textarea disabled={!canEdit} value={constraints} onChange={(e) => setConstraints(e.target.value)} rows={3} className={fieldClass} />
        </label>
        <label className={labelClass}>
          {t("crisp.business.risks")}
          <textarea disabled={!canEdit} value={risks} onChange={(e) => setRisks(e.target.value)} rows={3} className={fieldClass} />
        </label>
        <label className={labelClass}>
          {t("crisp.business.resources")}
          <textarea disabled={!canEdit} value={resources} onChange={(e) => setResources(e.target.value)} rows={3} className={fieldClass} />
        </label>
        <label className={labelClass}>
          {t("crisp.business.successCriteria")}
          <textarea disabled={!canEdit} value={successCriteria} onChange={(e) => setSuccessCriteria(e.target.value)} rows={3} className={fieldClass} />
        </label>
      </div>

      <label className={labelClass}>
        {t("crisp.business.expectedGoals")}
        <textarea disabled={!canEdit} value={expectedGoals} onChange={(e) => setExpectedGoals(e.target.value)} rows={2} className={fieldClass} />
      </label>

      <div>
        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-200">{t("crisp.business.kpis")}</h4>
        <p className="text-xs text-gray-500 dark:text-gray-400">{t("crisp.business.kpiNote")}</p>
        <div className="mt-2 overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="text-xs uppercase text-gray-500 dark:text-gray-400">
                <th className="pb-1 pr-2">{t("crisp.business.indicator")}</th>
                <th className="pb-1 pr-2">{t("crisp.business.baseline")}</th>
                <th className="pb-1 pr-2">{t("crisp.business.target")}</th>
                <th className="pb-1 pr-2">{t("crisp.business.actual")}</th>
                {canEdit && <th />}
              </tr>
            </thead>
            <tbody>
              {kpis.map((kpi, i) => (
                <tr key={i} className="border-t border-gray-200 dark:border-gray-800">
                  <td className="py-1 pr-2">
                    <input disabled={!canEdit} value={kpi.indicator} onChange={(e) => updateKpi(i, { indicator: e.target.value })} className={fieldClass} />
                  </td>
                  <td className="py-1 pr-2">
                    <input
                      disabled={!canEdit}
                      type="number"
                      value={kpi.baseline_value ?? ""}
                      onChange={(e) => updateKpi(i, { baseline_value: e.target.value === "" ? null : Number(e.target.value) })}
                      className={fieldClass}
                    />
                  </td>
                  <td className="py-1 pr-2">
                    <input
                      disabled={!canEdit}
                      type="number"
                      value={kpi.target_value ?? ""}
                      onChange={(e) => updateKpi(i, { target_value: e.target.value === "" ? null : Number(e.target.value) })}
                      className={fieldClass}
                    />
                  </td>
                  <td className="py-1 pr-2 text-xs text-gray-500 dark:text-gray-400">
                    {kpi.actual_value ?? t("crisp.business.actualPending")}
                  </td>
                  {canEdit && (
                    <td>
                      <button onClick={() => setKpis((prev) => prev.filter((_, idx) => idx !== i))} className="text-xs text-status-red">
                        ✕
                      </button>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
          {canEdit && (
            <button
              onClick={() => setKpis((prev) => [...prev, { indicator: "", baseline_value: null, target_value: null, actual_value: null }])}
              className="mt-2 rounded bg-gray-300 px-2 py-1 text-xs font-semibold text-gray-800 hover:bg-gray-400 dark:bg-gray-700 dark:text-gray-100"
            >
              + {t("crisp.business.addKpi")}
            </button>
          )}
        </div>
      </div>

      {canEdit ? (
        <button
          onClick={() => save.mutate()}
          disabled={save.isPending}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-500 disabled:opacity-50"
        >
          {save.isPending ? t("crisp.phase.save") + "..." : save.isSuccess ? t("crisp.phase.saved") : t("crisp.phase.save")}
        </button>
      ) : (
        <p className="text-xs text-gray-500 dark:text-gray-400">{t("crisp.phase.noPermission")}</p>
      )}
      {save.isError && (
        <p className="rounded-lg bg-red-100 p-2 text-xs text-red-700 ring-1 ring-red-300 dark:bg-red-950/40 dark:text-red-300 dark:ring-red-800/50">
          {extractErrorMessage(save.error, t("crisp.common.error"))}
        </p>
      )}
    </div>
  );
}
