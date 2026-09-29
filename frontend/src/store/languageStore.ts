import { create } from "zustand";

import type { Language } from "@/i18n/translations";

function readStored(): Language {
  try {
    const stored = localStorage.getItem("language");
    if (stored === "es" || stored === "en") return stored;
  } catch {
    // localStorage unavailable — fall through to browser language
  }
  return navigator.language?.toLowerCase().startsWith("en") ? "en" : "es";
}

interface LanguageState {
  language: Language;
  setLanguage: (language: Language) => void;
}

export const useLanguageStore = create<LanguageState>((set) => ({
  language: readStored(),
  setLanguage: (language) => {
    try {
      localStorage.setItem("language", language);
    } catch {
      // ignore
    }
    set({ language });
  },
}));
