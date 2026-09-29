import { useTranslation } from "@/i18n/useTranslation";
import type { Language } from "@/i18n/translations";
import { useLanguageStore } from "@/store/languageStore";

/** "Apartado de idiomas": a small labeled section to switch the app's language. */
export function LanguageSelector() {
  const language = useLanguageStore((s) => s.language);
  const setLanguage = useLanguageStore((s) => s.setLanguage);
  const { t } = useTranslation();

  return (
    <label className="flex items-center gap-1.5 whitespace-nowrap text-xs text-gray-600 dark:text-gray-300">
      <span className="hidden sm:inline">{t("language.label")}</span>
      <select
        value={language}
        onChange={(e) => setLanguage(e.target.value as Language)}
        aria-label={t("language.label")}
        className="rounded bg-gray-200 px-2 py-1 text-xs text-gray-700 hover:bg-gray-300 dark:bg-gray-800 dark:text-gray-200 dark:hover:bg-gray-700"
      >
        <option value="es">{t("language.es")}</option>
        <option value="en">{t("language.en")}</option>
      </select>
    </label>
  );
}
