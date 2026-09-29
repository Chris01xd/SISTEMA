import { useForm } from "react-hook-form";

import { useTranslation } from "@/i18n/useTranslation";
import type { EmergencyEventCreate } from "@/types";

interface ScenarioFormProps {
  hospitalId: string;
  onSubmit: (payload: EmergencyEventCreate) => void;
  isSubmitting?: boolean;
}

export function ScenarioForm({ hospitalId, onSubmit, isSubmitting }: ScenarioFormProps) {
  const { t } = useTranslation();
  const { register, handleSubmit, setValue, watch } = useForm<EmergencyEventCreate>({
    defaultValues: {
      hospital_id: hospitalId,
      name: t("scenarioForm.presetEarthquake"),
      event_type: "EARTHQUAKE",
      magnitude: 8.0,
      expected_victims: 120,
      severity_distribution: { minor: 0.4, delayed: 0.3, immediate: 0.2, critical: 0.08, expectant: 0.02 },
      arrival_curve: "surge_decay",
      arrival_window_minutes: 90,
    },
  });

  const EVENT_PRESETS: Record<string, Partial<EmergencyEventCreate>> = {
    [t("scenarioForm.presetEarthquake")]: {
      event_type: "EARTHQUAKE",
      magnitude: 8.0,
      expected_victims: 120,
      arrival_window_minutes: 90,
    },
    [t("scenarioForm.presetExplosion")]: {
      event_type: "EXPLOSION",
      magnitude: 0,
      expected_victims: 60,
      arrival_window_minutes: 45,
    },
    [t("scenarioForm.presetTransport")]: {
      event_type: "TRANSPORT",
      magnitude: 0,
      expected_victims: 35,
      arrival_window_minutes: 30,
    },
  };

  const applyPreset = (name: string) => {
    const preset = EVENT_PRESETS[name];
    setValue("name", name);
    if (preset.event_type) setValue("event_type", preset.event_type);
    if (preset.magnitude !== undefined) setValue("magnitude", preset.magnitude);
    if (preset.expected_victims) setValue("expected_victims", preset.expected_victims);
    if (preset.arrival_window_minutes) setValue("arrival_window_minutes", preset.arrival_window_minutes);
  };

  return (
    <form
      onSubmit={handleSubmit(onSubmit)}
      className="space-y-4 rounded-xl border border-gray-300 bg-gray-100 p-4 dark:border-gray-700 dark:bg-gray-900/60 sm:p-5"
    >
      <h3 className="text-sm font-semibold uppercase tracking-wide text-gray-600 dark:text-gray-300">
        {t("scenarioForm.title")}
      </h3>

      <div className="flex flex-wrap gap-2">
        {Object.keys(EVENT_PRESETS).map((name) => (
          <button
            type="button"
            key={name}
            onClick={() => applyPreset(name)}
            className="rounded-full bg-gray-200 px-3 py-1 text-xs text-gray-700 hover:bg-gray-300 dark:bg-gray-800 dark:text-gray-200 dark:hover:bg-gray-700"
          >
            {name}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <label className="flex flex-col text-sm text-gray-600 dark:text-gray-300">
          {t("scenarioForm.nameLabel")}
          <input
            {...register("name")}
            className="mt-1 rounded bg-gray-200 px-2 py-1.5 text-gray-900 dark:bg-gray-800 dark:text-white"
          />
        </label>

        <label className="flex flex-col text-sm text-gray-600 dark:text-gray-300">
          {t("scenarioForm.typeLabel")}
          <select
            {...register("event_type")}
            className="mt-1 rounded bg-gray-200 px-2 py-1.5 text-gray-900 dark:bg-gray-800 dark:text-white"
          >
            <option value="EARTHQUAKE">{t("scenarioForm.typeEarthquake")}</option>
            <option value="EXPLOSION">{t("scenarioForm.typeExplosion")}</option>
            <option value="TRANSPORT">{t("scenarioForm.typeTransport")}</option>
            <option value="INDUSTRIAL">{t("scenarioForm.typeIndustrial")}</option>
            <option value="OTHER">{t("scenarioForm.typeOther")}</option>
          </select>
        </label>

        <label className="flex flex-col text-sm text-gray-600 dark:text-gray-300">
          {t("scenarioForm.magnitudeLabel")}
          <input
            type="number"
            step="0.1"
            {...register("magnitude", { valueAsNumber: true })}
            className="mt-1 rounded bg-gray-200 px-2 py-1.5 text-gray-900 dark:bg-gray-800 dark:text-white"
          />
        </label>

        <label className="flex flex-col text-sm text-gray-600 dark:text-gray-300">
          {t("scenarioForm.victimsLabel")}
          <input
            type="number"
            {...register("expected_victims", { valueAsNumber: true })}
            className="mt-1 rounded bg-gray-200 px-2 py-1.5 text-gray-900 dark:bg-gray-800 dark:text-white"
          />
        </label>

        <label className="flex flex-col text-sm text-gray-600 dark:text-gray-300">
          {t("scenarioForm.arrivalCurveLabel")}
          <select
            {...register("arrival_curve")}
            className="mt-1 rounded bg-gray-200 px-2 py-1.5 text-gray-900 dark:bg-gray-800 dark:text-white"
          >
            <option value="surge_decay">{t("scenarioForm.curveSurge")}</option>
            <option value="linear">{t("scenarioForm.curveLinear")}</option>
            <option value="poisson_steady">{t("scenarioForm.curvePoisson")}</option>
          </select>
        </label>

        <label className="flex flex-col text-sm text-gray-600 dark:text-gray-300">
          {t("scenarioForm.windowLabel")}
          <input
            type="number"
            {...register("arrival_window_minutes", { valueAsNumber: true })}
            className="mt-1 rounded bg-gray-200 px-2 py-1.5 text-gray-900 dark:bg-gray-800 dark:text-white"
          />
        </label>
      </div>

      <fieldset className="grid grid-cols-2 gap-2 text-xs text-gray-600 dark:text-gray-300 sm:grid-cols-3 md:grid-cols-5">
        <legend className="mb-1 text-sm text-gray-600 dark:text-gray-300">{t("scenarioForm.severityLegend")}</legend>
        {(["minor", "delayed", "immediate", "critical", "expectant"] as const).map((key) => (
          <label key={key} className="flex flex-col">
            {key}
            <input
              type="number"
              step="0.01"
              {...register(`severity_distribution.${key}`, { valueAsNumber: true })}
              className="mt-1 rounded bg-gray-200 px-2 py-1 text-gray-900 dark:bg-gray-800 dark:text-white"
            />
          </label>
        ))}
      </fieldset>

      <p className="text-xs text-gray-500">
        {t("scenarioForm.preview")}: {watch("name")}
      </p>

      <button
        type="submit"
        disabled={isSubmitting}
        className="w-full rounded-lg bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-500 disabled:opacity-50 sm:w-auto"
      >
        {isSubmitting ? t("scenarioForm.creating") : t("scenarioForm.submit")}
      </button>
    </form>
  );
}
