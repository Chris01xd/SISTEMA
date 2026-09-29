import { useTranslation } from "@/i18n/useTranslation";
import { useThemeStore } from "@/store/themeStore";

/** Single button that flips between light ("claro") and dark ("noche") mode. */
export function ThemeToggle() {
  const theme = useThemeStore((s) => s.theme);
  const toggleTheme = useThemeStore((s) => s.toggleTheme);
  const { t } = useTranslation();
  const isDark = theme === "dark";

  return (
    <button
      onClick={toggleTheme}
      title={isDark ? t("theme.toggleToLight") : t("theme.toggleToDark")}
      aria-label={isDark ? t("theme.toggleToLight") : t("theme.toggleToDark")}
      className="flex items-center gap-1.5 whitespace-nowrap rounded bg-gray-200 px-2 py-1 text-xs text-gray-700 hover:bg-gray-300 dark:bg-gray-800 dark:text-gray-200 dark:hover:bg-gray-700"
    >
      {isDark ? (
        <svg viewBox="0 0 24 24" fill="currentColor" className="h-3.5 w-3.5">
          <path d="M12 4.5a1 1 0 0 1-1-1V2a1 1 0 1 1 2 0v1.5a1 1 0 0 1-1 1Zm0 15a1 1 0 0 1 1 1V22a1 1 0 1 1-2 0v-1.5a1 1 0 0 1 1-1ZM4.5 12a1 1 0 0 1-1 1H2a1 1 0 1 1 0-2h1.5a1 1 0 0 1 1 1Zm17.5 0a1 1 0 0 1-1 1h-1.5a1 1 0 1 1 0-2H21a1 1 0 0 1 1 1ZM6.34 6.34a1 1 0 0 1-1.41 0L3.87 5.29a1 1 0 0 1 1.42-1.42l1.05 1.06a1 1 0 0 1 0 1.41Zm12.02 12.02a1 1 0 0 1-1.41 0l-1.06-1.05a1 1 0 0 1 1.42-1.42l1.05 1.06a1 1 0 0 1 0 1.41ZM6.34 17.66a1 1 0 0 1 0 1.41l-1.05 1.06a1 1 0 0 1-1.42-1.42l1.06-1.05a1 1 0 0 1 1.41 0Zm12.02-12.02a1 1 0 0 1 0 1.41l-1.05 1.06a1 1 0 1 1-1.42-1.42l1.06-1.05a1 1 0 0 1 1.41 0ZM12 7a5 5 0 1 1 0 10 5 5 0 0 1 0-10Z" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" fill="currentColor" className="h-3.5 w-3.5">
          <path d="M21.75 15.5A9.75 9.75 0 1 1 8.5 2.25a.75.75 0 0 1 .82 1.13 7.5 7.5 0 0 0 10.3 10.3.75.75 0 0 1 1.13.82Z" />
        </svg>
      )}
      <span className="hidden sm:inline">{isDark ? t("theme.dark") : t("theme.light")}</span>
    </button>
  );
}
