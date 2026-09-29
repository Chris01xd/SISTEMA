import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { HospitalScene } from "@/components/DigitalTwin/HospitalScene";
import { useTwinWebSocket } from "@/hooks/useTwinWebSocket";
import { useTranslation } from "@/i18n/useTranslation";
import { api } from "@/services/api";
import { useAuthStore } from "@/store/authStore";
import type { HospitalTwinSnapshot } from "@/types";

interface TwinPageProps {
  hospitalId: string;
}

export function TwinPage({ hospitalId }: TwinPageProps) {
  useTwinWebSocket();
  const queryClient = useQueryClient();
  const token = useAuthStore((s) => s.token);
  const { t } = useTranslation();

  const { data, isLoading, error } = useQuery({
    queryKey: ["twin-snapshot", hospitalId],
    queryFn: async () => {
      const res = await api.get<HospitalTwinSnapshot>(`/hospitals/${hospitalId}/twin-snapshot`);
      return res.data;
    },
    enabled: Boolean(hospitalId),
  });

  const seedLayout = useMutation({
    mutationFn: async () => {
      await api.post(`/hospitals/${hospitalId}/seed-default-layout`, {});
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["twin-snapshot", hospitalId] }),
  });

  if (!hospitalId) {
    return (
      <p className="p-6 text-gray-500 dark:text-gray-400">
        {t("twin.needHospitalPrefix")} <strong>{t("twin.needHospitalButton")}</strong> {t("twin.needHospitalSuffix")}
      </p>
    );
  }
  if (isLoading) return <p className="p-6 text-gray-500 dark:text-gray-400">{t("twin.loading")}</p>;
  if (error) {
    return <p className="p-6 text-red-600 dark:text-red-400">{t("twin.notFound")}</p>;
  }
  if (!data) return null;

  if (data.floors.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4 p-6 text-center">
        <p className="max-w-md text-gray-500 dark:text-gray-400">
          <strong>{data.hospital.name}</strong> {t("twin.emptyLayoutSuffix")}
        </p>
        {token ? (
          <button
            onClick={() => seedLayout.mutate()}
            disabled={seedLayout.isPending}
            className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-500 disabled:opacity-50"
          >
            {seedLayout.isPending ? t("twin.generatingLayout") : t("twin.generateLayout")}
          </button>
        ) : (
          <p className="text-xs text-gray-500">{t("twin.loginToGenerate")}</p>
        )}
        {seedLayout.isError && <p className="text-xs text-red-600 dark:text-red-400">{t("twin.generateError")}</p>}
      </div>
    );
  }

  return (
    <div className="h-full w-full">
      <HospitalScene snapshot={data} />
    </div>
  );
}
