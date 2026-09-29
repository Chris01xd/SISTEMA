import { useMutation } from "@tanstack/react-query";
import { useState } from "react";

import { useTranslation } from "@/i18n/useTranslation";
import { api } from "@/services/api";

interface ReportsPageProps {
  simulationId: string;
}

const FORMATS = ["DOCX", "PDF", "XLSX"] as const;

export function ReportsPage({ simulationId }: ReportsPageProps) {
  const [lastReportUrl, setLastReportUrl] = useState<string | null>(null);
  const { t } = useTranslation();

  function extractErrorMessage(err: unknown): string {
    const anyErr = err as { response?: { status?: number; data?: { detail?: string } } };
    if (anyErr?.response?.status === 401 || anyErr?.response?.status === 403) {
      return t("reports.errorAuth");
    }
    if (anyErr?.response?.status === 404) return t("reports.errorNotFound");
    const detail = anyErr?.response?.data?.detail;
    return typeof detail === "string" ? detail : t("reports.errorGeneric");
  }

  const generate = useMutation({
    mutationFn: async (format: (typeof FORMATS)[number]) => {
      const res = await api.post("/reports/generate", { simulation_id: simulationId, format });
      return res.data;
    },
    onSuccess: (data) => setLastReportUrl(`/api/v1/reports/${data.id}/download`),
  });

  return (
    <div className="space-y-4 p-4 sm:p-6">
      <h2 className="text-xl font-bold text-gray-900 dark:text-white">{t("reports.title")}</h2>
      <p className="break-all text-sm text-gray-500 dark:text-gray-400">
        {t("reports.simulationLabel")}: {simulationId || t("reports.noneSelected")}
      </p>

      <div className="flex flex-wrap gap-3">
        {FORMATS.map((fmt) => (
          <button
            key={fmt}
            disabled={!simulationId || generate.isPending}
            onClick={() => generate.mutate(fmt)}
            className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {generate.isPending && generate.variables === fmt ? t("reports.generating") : `${t("reports.generate")} ${fmt}`}
          </button>
        ))}
      </div>

      {generate.isError && (
        <p className="rounded-lg bg-red-100 p-3 text-xs text-red-700 ring-1 ring-red-300 dark:bg-red-950/40 dark:text-red-300 dark:ring-red-800/50">
          {extractErrorMessage(generate.error)}
        </p>
      )}

      {lastReportUrl && (
        <a href={lastReportUrl} className="inline-block text-sm text-emerald-600 underline dark:text-emerald-400">
          {t("reports.downloadLast")}
        </a>
      )}
    </div>
  );
}
