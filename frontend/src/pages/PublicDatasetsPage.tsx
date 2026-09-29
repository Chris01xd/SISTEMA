import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { useTranslation } from "@/i18n/useTranslation";
import { extractErrorMessage } from "@/lib/errorMessage";
import {
  type CatalogEntry,
  getDatasetPreview,
  getDatasetSummary,
  ingestDataset,
  listCatalog,
} from "@/services/datasetsApi";

// SeverityLevel colors: 1 minor .. 4 critical (same scale as backend/app/models/clinical.py).
const SEVERITY_COLOR: Record<string, string> = {
  "1": "bg-status-green",
  "2": "bg-status-yellow",
  "3": "bg-orange-500",
  "4": "bg-status-red",
};

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-gray-100 p-3 dark:bg-gray-900">
      <p className="text-xs text-gray-500 dark:text-gray-400">{label}</p>
      <p className="mt-1 text-lg font-bold text-gray-900 dark:text-white">{value}</p>
    </div>
  );
}

function DatasetCard({ entry }: { entry: CatalogEntry }) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const datasetId = entry.dataset_id;

  const ingest = useMutation({
    mutationFn: () => ingestDataset(entry.key),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["dataset-catalog"] });
      queryClient.invalidateQueries({ queryKey: ["dataset-summary"] });
      queryClient.invalidateQueries({ queryKey: ["dataset-preview"] });
    },
  });

  const summary = useQuery({
    queryKey: ["dataset-summary", datasetId],
    queryFn: () => getDatasetSummary(datasetId!),
    enabled: !!datasetId,
  });
  const preview = useQuery({
    queryKey: ["dataset-preview", datasetId],
    queryFn: () => getDatasetPreview(datasetId!),
    enabled: !!datasetId,
  });

  const stats = summary.data;
  const severityTotal = Object.values(stats?.severity_distribution ?? {}).reduce((a, b) => a + b, 0);
  const hourData = (stats?.admissions_by_hour ?? []).map((count, hour) => ({ hour, count }));

  function errorText(err: unknown): string {
    const status = (err as { response?: { status?: number } })?.response?.status;
    if (status === 401 || status === 403) return t("datasets.errorAuth");
    return extractErrorMessage(err, t("datasets.errorGeneric"));
  }

  return (
    <section className="space-y-4 rounded-xl border border-gray-200 p-4 dark:border-gray-800">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 space-y-1">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            {entry.name} <span className="text-sm font-normal text-gray-500">v{entry.version}</span>
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-300">{entry.description}</p>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            {t("datasets.license")}: {entry.license} ·{" "}
            <a href={entry.homepage} target="_blank" rel="noreferrer" className="underline">
              {t("datasets.source")}
            </a>
          </p>
          <p className="text-xs italic text-gray-500 dark:text-gray-400">
            {t("datasets.citation")}: {entry.citation}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className={`text-xs font-semibold ${entry.ingested ? "text-status-green" : "text-gray-500"}`}>
            {entry.ingested ? t("datasets.imported") : t("datasets.notImported")}
          </span>
          <button
            disabled={ingest.isPending}
            onClick={() => ingest.mutate()}
            className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {ingest.isPending ? t("datasets.importing") : entry.ingested ? t("datasets.reimport") : t("datasets.import")}
          </button>
        </div>
      </div>

      {ingest.isError && (
        <p className="rounded-lg bg-red-100 p-3 text-xs text-red-700 ring-1 ring-red-300 dark:bg-red-950/40 dark:text-red-300 dark:ring-red-800/50">
          {errorText(ingest.error)}
        </p>
      )}

      {datasetId && summary.isLoading && <p className="text-sm text-gray-500">{t("datasets.loadingSummary")}</p>}

      {stats && stats.rows > 0 && (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
            <Stat label={t("datasets.rows")} value={String(stats.rows)} />
            <Stat label={t("datasets.patients")} value={String(stats.patients)} />
            <Stat label={t("datasets.age")} value={String(stats.age?.mean)} />
            <Stat label={t("datasets.los")} value={`${stats.los_hours?.median} ${t("datasets.hours")}`} />
            <Stat label={t("datasets.icuRate")} value={`${stats.icu_rate_pct}%`} />
            <Stat label={t("datasets.mortality")} value={`${stats.mortality_rate_pct}%`} />
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <div>
              <h4 className="mb-2 text-sm font-semibold text-gray-900 dark:text-white">{t("datasets.severityTitle")}</h4>
              <div className="space-y-2">
                {Object.entries(stats.severity_distribution ?? {}).map(([level, count]) => (
                  <div key={level} className="flex items-center gap-2 text-xs text-gray-700 dark:text-gray-300">
                    <span className="w-24 shrink-0">{t(`datasets.severityLabels.${level}`)}</span>
                    <div className="h-3 flex-1 overflow-hidden rounded bg-gray-200 dark:bg-gray-800">
                      <div
                        className={`h-full ${SEVERITY_COLOR[level] ?? "bg-gray-500"}`}
                        style={{ width: `${(100 * count) / severityTotal}%` }}
                      />
                    </div>
                    <span className="w-8 text-right tabular-nums">{count}</span>
                  </div>
                ))}
              </div>
              <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                {stats.severity_coverage_pct}% {t("datasets.severityCoverage")}
              </p>
            </div>

            <div>
              <h4 className="mb-2 text-sm font-semibold text-gray-900 dark:text-white">{t("datasets.hourTitle")}</h4>
              <div className="h-40">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={hourData}>
                    <XAxis dataKey="hour" tick={{ fontSize: 10 }} interval={2} />
                    <YAxis tick={{ fontSize: 10 }} width={28} allowDecimals={false} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#6366f1" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div>
            <h4 className="mb-2 text-sm font-semibold text-gray-900 dark:text-white">{t("datasets.diagnosesTitle")}</h4>
            <ul className="space-y-1 text-xs text-gray-700 dark:text-gray-300">
              {stats.top_diagnoses?.slice(0, 5).map((d) => (
                <li key={d.diagnosis} className="flex justify-between gap-3">
                  <span className="truncate">{d.diagnosis}</span>
                  <span className="tabular-nums">{d.count}</span>
                </li>
              ))}
            </ul>
          </div>

          {preview.data && (
            <div>
              <h4 className="mb-2 text-sm font-semibold text-gray-900 dark:text-white">{t("datasets.previewTitle")}</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs text-gray-700 dark:text-gray-300">
                  <thead>
                    <tr className="border-b border-gray-200 dark:border-gray-800">
                      {preview.data.columns.map((c) => (
                        <th key={c} className="whitespace-nowrap px-2 py-1 font-semibold">
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.data.rows.map((row, i) => (
                      <tr key={i} className="border-b border-gray-100 dark:border-gray-900">
                        {preview.data.columns.map((c) => (
                          <td key={c} className="max-w-[16rem] truncate whitespace-nowrap px-2 py-1">
                            {row[c] === null || row[c] === undefined ? "—" : String(row[c])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <p className="text-xs text-gray-500 dark:text-gray-400">{t("datasets.periodNote")}</p>
        </>
      )}
    </section>
  );
}

export function PublicDatasetsPage() {
  const { t } = useTranslation();
  const catalog = useQuery({ queryKey: ["dataset-catalog"], queryFn: listCatalog });

  return (
    <div className="space-y-4 p-4 sm:p-6">
      <div>
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">{t("datasets.title")}</h2>
        <p className="text-sm text-gray-500 dark:text-gray-400">{t("datasets.subtitle")}</p>
      </div>

      {catalog.isError && (
        <p className="rounded-lg bg-red-100 p-3 text-xs text-red-700 ring-1 ring-red-300 dark:bg-red-950/40 dark:text-red-300 dark:ring-red-800/50">
          {t("datasets.errorAuth")}
        </p>
      )}

      {catalog.data?.map((entry) => <DatasetCard key={entry.key} entry={entry} />)}
    </div>
  );
}
