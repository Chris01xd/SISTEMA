import { useQuery } from "@tanstack/react-query";

import { useTranslation } from "@/i18n/useTranslation";
import { getAuditLog } from "@/services/crispApi";

interface Props {
  projectId: string;
}

export function AuditHistoryPanel({ projectId }: Props) {
  const { t } = useTranslation();
  const audit = useQuery({ queryKey: ["crisp-audit", projectId], queryFn: () => getAuditLog(projectId) });

  return (
    <div className="space-y-3">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{t("crisp.audit.title")}</h3>
      <div className="overflow-x-auto rounded-lg border border-gray-300 dark:border-gray-700">
        <table className="w-full text-left text-xs">
          <thead className="bg-gray-200 text-gray-600 dark:bg-gray-800 dark:text-gray-300">
            <tr>
              <th className="px-2 py-1.5">{t("crisp.audit.time")}</th>
              <th className="px-2 py-1.5">{t("crisp.audit.action")}</th>
              <th className="px-2 py-1.5">{t("crisp.audit.entity")}</th>
            </tr>
          </thead>
          <tbody>
            {audit.data?.map((entry) => (
              <tr key={entry.id} className="border-t border-gray-200 dark:border-gray-800">
                <td className="whitespace-nowrap px-2 py-1.5 text-gray-500 dark:text-gray-400">
                  {new Date(entry.occurred_at).toLocaleString()}
                </td>
                <td className="px-2 py-1.5 font-medium text-gray-800 dark:text-gray-200">{entry.action}</td>
                <td className="px-2 py-1.5 text-gray-600 dark:text-gray-300">
                  {entry.entity_type} · {entry.entity_id.slice(0, 8)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {audit.data?.length === 0 && <p className="p-3 text-xs text-gray-500 dark:text-gray-400">{t("crisp.audit.empty")}</p>}
      </div>
    </div>
  );
}
