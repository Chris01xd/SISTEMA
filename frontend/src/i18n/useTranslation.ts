import { useLanguageStore } from "@/store/languageStore";

import { translations } from "./translations";

function getPath(obj: unknown, path: string): string {
  const value = path.split(".").reduce<unknown>((acc, key) => {
    if (acc && typeof acc === "object" && key in acc) return (acc as Record<string, unknown>)[key];
    return undefined;
  }, obj);
  return typeof value === "string" ? value : path;
}

export function useTranslation() {
  const language = useLanguageStore((s) => s.language);
  const t = (key: string) => getPath(translations[language], key);
  return { t, language };
}
