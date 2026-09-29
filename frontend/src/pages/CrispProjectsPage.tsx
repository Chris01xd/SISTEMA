import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { useTranslation } from "@/i18n/useTranslation";
import { extractErrorMessage } from "@/lib/errorMessage";
import { createProject, listLoadingUnits, listProjects } from "@/services/crispApi";

const STATUS_ACCENT: Record<string, string> = {
  NOT_STARTED: "text-gray-500",
  IN_PROGRESS: "text-status-yellow",
  ON_HOLD: "text-amber-600",
  COMPLETED: "text-status-green",
  CANCELLED: "text-status-red",
};

export function CrispProjectsPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [startDate, setStartDate] = useState(new Date().toISOString().slice(0, 10));
  const [unitIds, setUnitIds] = useState<string[]>([]);
  const [showForm, setShowForm] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const projects = useQuery({ queryKey: ["crisp-projects"], queryFn: listProjects });
  const units = useQuery({ queryKey: ["mining-loading-units"], queryFn: listLoadingUnits });

  const create = useMutation({
    mutationFn: () => createProject({ name, description, start_date: startDate, loading_unit_ids: unitIds }),
    onSuccess: (project) => {
      queryClient.invalidateQueries({ queryKey: ["crisp-projects"] });
      navigate(`/crisp/${project.id}`);
    },
    onError: (err) => setError(extractErrorMessage(err, t("crisp.common.error"))),
  });

  const fieldClass = "mt-1 rounded bg-gray-200 px-2 py-1.5 text-sm text-gray-900 dark:bg-gray-800 dark:text-white";

  return (
    <div className="space-y-4 p-4 sm:p-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">{t("crisp.projects.title")}</h2>
        <button
          onClick={() => setShowForm((s) => !s)}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-500"
        >
          + {t("crisp.projects.create")}
        </button>
      </div>

      {showForm && (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            create.mutate();
          }}
          className="grid grid-cols-1 gap-3 rounded-xl border border-gray-300 bg-gray-100 p-4 dark:border-gray-700 dark:bg-gray-900/60 sm:grid-cols-2"
        >
          <label className="flex flex-col text-sm text-gray-600 dark:text-gray-300">
            {t("crisp.projects.name")}
            <input value={name} onChange={(e) => setName(e.target.value)} className={fieldClass} required />
          </label>
          <label className="flex flex-col text-sm text-gray-600 dark:text-gray-300">
            {t("crisp.projects.startDate")}
            <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className={fieldClass} required />
          </label>
          <label className="flex flex-col text-sm text-gray-600 dark:text-gray-300 sm:col-span-2">
            {t("crisp.projects.description")}
            <textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={2} className={fieldClass} />
          </label>
          <label className="flex flex-col text-sm text-gray-600 dark:text-gray-300 sm:col-span-2">
            {t("crisp.projects.loadingUnits")}
            <select
              multiple
              value={unitIds}
              onChange={(e) => setUnitIds(Array.from(e.target.selectedOptions, (o) => o.value))}
              className={fieldClass}
            >
              {units.data?.map((u) => (
                <option key={u.id} value={u.id}>
                  {u.code} — {u.brand} {u.model}
                </option>
              ))}
            </select>
          </label>
          <button
            type="submit"
            disabled={!name || create.isPending}
            className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-500 disabled:opacity-50 sm:col-span-2 sm:w-fit"
          >
            {create.isPending ? t("crisp.projects.creating") : t("crisp.projects.submit")}
          </button>
        </form>
      )}

      {error && (
        <p className="rounded-lg bg-red-100 p-3 text-xs text-red-700 ring-1 ring-red-300 dark:bg-red-950/40 dark:text-red-300 dark:ring-red-800/50">
          {error}
        </p>
      )}

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {projects.data?.map((project) => (
          <Link
            key={project.id}
            to={`/crisp/${project.id}`}
            className="rounded-xl border border-gray-300 bg-gray-100 p-4 hover:border-indigo-500 dark:border-gray-700 dark:bg-gray-900/60"
          >
            <p className="font-semibold text-gray-900 dark:text-white">{project.name}</p>
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              {t("crisp.projects.phase")} {project.current_phase_number}/6 · {t("crisp.projects.progress")} {project.progress_percent}%
            </p>
            <p className={`mt-1 text-xs font-semibold ${STATUS_ACCENT[project.overall_status]}`}>{project.overall_status}</p>
          </Link>
        ))}
        {projects.data?.length === 0 && <p className="text-sm text-gray-500 dark:text-gray-400">{t("crisp.projects.empty")}</p>}
      </div>
    </div>
  );
}
