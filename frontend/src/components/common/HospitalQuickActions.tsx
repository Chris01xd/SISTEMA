import { useState } from "react";

import { useTranslation } from "@/i18n/useTranslation";
import { api } from "@/services/api";
import { useAuthStore } from "@/store/authStore";

interface HospitalQuickActionsProps {
  onHospitalReady: (hospitalId: string) => void;
}

/** One-click "create a demo hospital with a full ER/ICU/Surgery/Ward layout"
 * action, so a new user can see the 3D twin populated immediately instead of
 * having to know a hospital ID or build floors/rooms/beds by hand. */
export function HospitalQuickActions({ onHospitalReady }: HospitalQuickActionsProps) {
  const token = useAuthStore((s) => s.token);
  const { t } = useTranslation();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const createDemoHospital = async () => {
    setBusy(true);
    setError(null);
    try {
      const defaultName = t("quickActions.namePromptDefault");
      const name = window.prompt(t("quickActions.namePromptTitle"), defaultName) || defaultName;
      const res = await api.post("/hospitals", { name, location: t("quickActions.location"), total_capacity: 200 });
      const hospitalId = res.data.id as string;
      await api.post(`/hospitals/${hospitalId}/seed-default-layout`, {});
      onHospitalReady(hospitalId);
    } catch {
      setError(t("quickActions.error"));
    } finally {
      setBusy(false);
    }
  };

  if (!token) {
    return (
      <span className="hidden text-xs text-gray-500 lg:inline">{t("quickActions.loginPrompt")}</span>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      <button
        onClick={createDemoHospital}
        disabled={busy}
        className="whitespace-nowrap rounded bg-indigo-600 px-3 py-1 text-xs font-semibold text-white hover:bg-indigo-500 disabled:opacity-50"
      >
        {busy ? t("quickActions.creating") : t("quickActions.create")}
      </button>
      {error && <span className="text-xs text-red-600 dark:text-red-400">{error}</span>}
    </div>
  );
}
