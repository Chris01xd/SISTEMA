import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { KpiCard } from "@/components/Dashboard/KpiCard";
import { useTranslation } from "@/i18n/useTranslation";
import { listAlerts, listDeployments, listLoadingUnits, listProjects } from "@/services/crispApi";

const SEVERITY_STYLE: Record<string, string> = {
  CRITICAL: "bg-red-100 text-red-800 ring-1 ring-red-300 dark:bg-red-950/40 dark:text-red-200 dark:ring-red-800/50",
  WARNING: "bg-amber-100 text-amber-800 ring-1 ring-amber-300 dark:bg-amber-950/40 dark:text-amber-200 dark:ring-amber-800/50",
  NORMAL: "bg-gray-100 text-gray-700 ring-1 ring-gray-300 dark:bg-gray-900 dark:text-gray-300 dark:ring-gray-700",
  OUT_OF_SERVICE: "bg-gray-300 text-gray-800 ring-1 ring-gray-400 dark:bg-gray-800 dark:text-gray-200 dark:ring-gray-600",
};

export function MiningDashboardPage() {
  const { t } = useTranslation();

  const projects = useQuery({ queryKey: ["crisp-projects"], queryFn: listProjects });
  const units = useQuery({ queryKey: ["mining-loading-units"], queryFn: listLoadingUnits });
  const alerts = useQuery({ queryKey: ["mining-alerts-recent"], queryFn: () => listAlerts() });

  const projectIds = projects.data?.map((p) => p.id) ?? [];
  const deployments = useQuery({
    queryKey: ["crisp-all-deployments", projectIds.join(",")],
    queryFn: async () => (await Promise.all(projectIds.map((id) => listDeployments(id)))).flat(),
    enabled: projectIds.length > 0,
  });

  const activeProjects = projects.data?.filter((p) => p.overall_status === "IN_PROGRESS").length ?? 0;
  const activeDeployments = deployments.data?.filter((d) => d.service_status === "ACTIVE").length ?? 0;
  const today = new Date().toISOString().slice(0, 10);
  const needsRetraining = deployments.data?.filter((d) => d.next_retraining_date && d.next_retraining_date <= today).length ?? 0;
  const recentAlerts = (alerts.data ?? []).slice(0, 8);

  return (
    <div className="space-y-4 p-4 sm:space-y-6 sm:p-6">
      <h2 className="text-xl font-bold text-gray-900 dark:text-white">{t("crisp.dashboard.title")}</h2>

      <div className="grid grid-cols-2 gap-3 sm:gap-4 md:grid-cols-4">
        <KpiCard label={t("crisp.dashboard.activeProjects")} value={activeProjects} />
        <KpiCard label={t("crisp.dashboard.monitoredUnits")} value={units.data?.length ?? 0} />
        <KpiCard label={t("crisp.dashboard.deployedModels")} value={activeDeployments} accent="green" />
        <KpiCard label={t("crisp.dashboard.needsRetraining")} value={needsRetraining} accent={needsRetraining > 0 ? "red" : "neutral"} />
      </div>

      <div className="rounded-xl border border-gray-300 bg-gray-100 p-4 dark:border-gray-700 dark:bg-gray-900/60">
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-gray-600 dark:text-gray-300">
          {t("crisp.dashboard.recentAlerts")}
        </h3>
        {recentAlerts.length === 0 ? (
          <p className="text-sm text-gray-500 dark:text-gray-400">{t("crisp.dashboard.empty")}</p>
        ) : (
          <ul className="space-y-2">
            {recentAlerts.map((alert) => (
              <li key={alert.id} className={`rounded-lg p-3 text-sm ${SEVERITY_STYLE[alert.severity]}`}>
                <span className="font-semibold">{alert.severity}</span> — {alert.message}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {projects.data?.map((project) => (
          <Link
            key={project.id}
            to={`/crisp/${project.id}`}
            className="rounded-xl border border-gray-300 bg-gray-100 p-4 hover:border-indigo-500 dark:border-gray-700 dark:bg-gray-900/60"
          >
            <p className="font-semibold text-gray-900 dark:text-white">{project.name}</p>
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              {t("crisp.projects.phase")} {project.current_phase_number}/6 · {project.progress_percent}%
            </p>
          </Link>
        ))}
      </div>
    </div>
  );
}
