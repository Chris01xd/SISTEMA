import { useMutation } from "@tanstack/react-query";

import { useTranslation } from "@/i18n/useTranslation";
import { extractErrorMessage } from "@/lib/errorMessage";
import { api } from "@/services/api";

interface Props {
  projectId: string;
}

/** The report endpoint requires auth (`crisp:report:download`), so a plain
 * <a href> can't carry the JWT header — fetch as a blob and trigger the save
 * via a throwaway object URL instead. */
export function ReportsPanel({ projectId }: Props) {
  const { t } = useTranslation();

  const download = useMutation({
    mutationFn: async (fmt: "pdf" | "xlsx") => {
      const res = await api.get(`/crisp/projects/${projectId}/report`, { params: { fmt }, responseType: "blob" });
      const url = URL.createObjectURL(res.data as Blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `crisp_report_${projectId}.${fmt}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    },
  });

  return (
    <div className="space-y-3">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{t("crisp.reports.title")}</h3>
      <div className="flex flex-wrap gap-3">
        <button
          onClick={() => download.mutate("pdf")}
          disabled={download.isPending}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-500 disabled:opacity-50"
        >
          {t("crisp.reports.downloadPdf")}
        </button>
        <button
          onClick={() => download.mutate("xlsx")}
          disabled={download.isPending}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-500 disabled:opacity-50"
        >
          {t("crisp.reports.downloadXlsx")}
        </button>
      </div>
      {download.isError && (
        <p className="rounded-lg bg-red-100 p-2 text-xs text-red-700 ring-1 ring-red-300 dark:bg-red-950/40 dark:text-red-300 dark:ring-red-800/50">
          {extractErrorMessage(download.error, t("crisp.common.error"))}
        </p>
      )}
    </div>
  );
}
