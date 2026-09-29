import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useParams } from "react-router-dom";

import { AuditHistoryPanel } from "@/components/Crisp/AuditHistoryPanel";
import { BusinessUnderstandingPanel } from "@/components/Crisp/BusinessUnderstandingPanel";
import { DataPreparationPanel } from "@/components/Crisp/DataPreparationPanel";
import { DataUnderstandingPanel } from "@/components/Crisp/DataUnderstandingPanel";
import { DeploymentPanel } from "@/components/Crisp/DeploymentPanel";
import { EvaluationPanel } from "@/components/Crisp/EvaluationPanel";
import { ModelingPanel } from "@/components/Crisp/ModelingPanel";
import { MonitoringPanel } from "@/components/Crisp/MonitoringPanel";
import { PhaseStepper } from "@/components/Crisp/PhaseStepper";
import { ReportsPanel } from "@/components/Crisp/ReportsPanel";
import { useTranslation } from "@/i18n/useTranslation";
import { getProject, listPhases } from "@/services/crispApi";
import { useAuthStore } from "@/store/authStore";

type TabKey = "business" | "data" | "prep" | "modeling" | "evaluation" | "deployment" | "monitoring" | "audit" | "reports";

const PHASE_TO_TAB: Record<number, TabKey> = { 1: "business", 2: "data", 3: "prep", 4: "modeling", 5: "evaluation", 6: "deployment" };
const TAB_TO_PHASE: Partial<Record<TabKey, number>> = Object.fromEntries(
  Object.entries(PHASE_TO_TAB).map(([phaseNumber, tabKey]) => [tabKey, Number(phaseNumber)])
);

export function CrispProjectDetailPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const { t } = useTranslation();
  const token = useAuthStore((s) => s.token);
  const [tab, setTab] = useState<TabKey>("business");

  const project = useQuery({ queryKey: ["crisp-project", projectId], queryFn: () => getProject(projectId!), enabled: !!projectId });
  const phases = useQuery({ queryKey: ["crisp-phases", projectId], queryFn: () => listPhases(projectId!), enabled: !!projectId });

  if (!projectId) return null;
  const canEdit = !!token;

  const tabs: { key: TabKey; label: string }[] = [
    { key: "business", label: t("crisp.business.title") },
    { key: "data", label: t("crisp.data.title") },
    { key: "prep", label: t("crisp.prep.title") },
    { key: "modeling", label: t("crisp.modeling.title") },
    { key: "evaluation", label: t("crisp.evaluation.title") },
    { key: "deployment", label: t("crisp.deployment.title") },
    { key: "monitoring", label: t("crisp.monitoring.title") },
    { key: "audit", label: t("crisp.audit.title") },
    { key: "reports", label: t("crisp.reports.title") },
  ];

  const activePhase = phases.data?.find((p) => PHASE_TO_TAB[p.phase_number] === tab);

  return (
    <div className="space-y-5 p-4 sm:p-6">
      <Link to="/crisp/projects" className="text-sm text-indigo-600 dark:text-indigo-400">
        {t("crisp.projects.backToList")}
      </Link>

      {project.data && (
        <div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-white">{project.data.name}</h2>
          <p className="text-sm text-gray-500 dark:text-gray-400">{project.data.description}</p>
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            {t("crisp.projects.progress")}: {project.data.progress_percent}% · {project.data.overall_status}
          </p>
        </div>
      )}

      {phases.data && (
        <PhaseStepper
          projectId={projectId}
          phases={phases.data}
          activePhaseNumber={TAB_TO_PHASE[tab] ?? 1}
          onSelectPhase={(n) => setTab(PHASE_TO_TAB[n] ?? "business")}
        />
      )}

      <div className="flex flex-wrap gap-1 border-b border-gray-300 dark:border-gray-700">
        {tabs.map((tabDef) => (
          <button
            key={tabDef.key}
            onClick={() => setTab(tabDef.key)}
            className={`rounded-t-lg px-3 py-1.5 text-xs font-semibold ${
              tab === tabDef.key
                ? "bg-gray-100 text-gray-900 dark:bg-gray-900 dark:text-white"
                : "text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            }`}
          >
            {tabDef.label}
          </button>
        ))}
      </div>

      <div className="rounded-b-xl rounded-tr-xl border border-gray-300 bg-gray-50 p-4 dark:border-gray-700 dark:bg-gray-950/40">
        {tab === "business" && activePhase && <BusinessUnderstandingPanel projectId={projectId} phase={activePhase} canEdit={canEdit} />}
        {tab === "data" && <DataUnderstandingPanel projectId={projectId} canEdit={canEdit} />}
        {tab === "prep" && <DataPreparationPanel projectId={projectId} canEdit={canEdit} />}
        {tab === "modeling" && <ModelingPanel projectId={projectId} datasetVersionId="" canEdit={canEdit} />}
        {tab === "evaluation" && <EvaluationPanel projectId={projectId} canEdit={canEdit} />}
        {tab === "deployment" && <DeploymentPanel projectId={projectId} canEdit={canEdit} />}
        {tab === "monitoring" && <MonitoringPanel projectId={projectId} />}
        {tab === "audit" && <AuditHistoryPanel projectId={projectId} />}
        {tab === "reports" && <ReportsPanel projectId={projectId} />}
      </div>
    </div>
  );
}
