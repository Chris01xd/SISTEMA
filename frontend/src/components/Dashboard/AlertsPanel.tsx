import { useTranslation } from "@/i18n/useTranslation";
import type { AIRecommendationMessage } from "@/types";

interface AlertsPanelProps {
  recommendations: AIRecommendationMessage[];
}

export function AlertsPanel({ recommendations }: AlertsPanelProps) {
  const { t } = useTranslation();
  return (
    <div className="rounded-xl border border-gray-300 bg-gray-100 p-4 dark:border-gray-700 dark:bg-gray-900/60">
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-gray-600 dark:text-gray-300">
        {t("alerts.title")}
      </h3>
      {recommendations.length === 0 ? (
        <p className="text-sm text-gray-500">{t("alerts.empty")}</p>
      ) : (
        <ul className="space-y-2">
          {recommendations.map((rec, i) => (
            <li
              key={i}
              className="rounded-lg bg-amber-100 p-3 text-sm text-amber-800 ring-1 ring-amber-300 dark:bg-amber-950/40 dark:text-amber-200 dark:ring-amber-800/50"
            >
              {rec.message}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
