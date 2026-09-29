import { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";

import { AuthBar } from "@/components/common/AuthBar";
import { HospitalQuickActions } from "@/components/common/HospitalQuickActions";
import { LanguageSelector } from "@/components/common/LanguageSelector";
import { ThemeToggle } from "@/components/common/ThemeToggle";
import { useTranslation } from "@/i18n/useTranslation";
import { CrispProjectDetailPage } from "@/pages/CrispProjectDetailPage";
import { CrispProjectsPage } from "@/pages/CrispProjectsPage";
import { DashboardPage } from "@/pages/DashboardPage";
import { MiningDashboardPage } from "@/pages/MiningDashboardPage";
import { ModelComparisonPage } from "@/pages/ModelComparisonPage";
import { PublicDatasetsPage } from "@/pages/PublicDatasetsPage";
import { ReportsPage } from "@/pages/ReportsPage";
import { SimulatorPage } from "@/pages/SimulatorPage";
import { TwinPage } from "@/pages/TwinPage";
import { useLanguageStore } from "@/store/languageStore";
import { useThemeStore } from "@/store/themeStore";

export default function App() {
  const [hospitalId, setHospitalId] = useState("");
  const [simulationId, setSimulationId] = useState("");
  const { t } = useTranslation();
  const theme = useThemeStore((s) => s.theme);
  const language = useLanguageStore((s) => s.language);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  useEffect(() => {
    document.documentElement.lang = language;
  }, [language]);

  const NAV_ITEMS = [
    { to: "/", label: t("nav.dashboard") },
    { to: "/twin", label: t("nav.twin") },
    { to: "/simulator", label: t("nav.simulator") },
    { to: "/reports", label: t("nav.reports") },
    { to: "/datasets", label: t("nav.datasets") },
    { to: "/ml", label: t("nav.ml") },
    { to: "/crisp", label: t("crisp.navDashboard") },
    { to: "/crisp/projects", label: t("crisp.navProjects") },
  ];

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-white dark:bg-gray-950">
      <header className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 border-b border-gray-200 px-4 py-2 dark:border-gray-800 sm:px-6">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
          <h1 className="whitespace-nowrap text-sm font-bold uppercase tracking-widest text-gray-900 dark:text-gray-100">
            {t("app.title")} <span className="hidden text-gray-500 sm:inline">{t("app.subtitle")}</span>
          </h1>
          <nav className="flex flex-wrap gap-x-4 gap-y-1">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `whitespace-nowrap text-sm ${
                    isActive
                      ? "font-semibold text-gray-900 dark:text-white"
                      : "text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <HospitalQuickActions onHospitalReady={setHospitalId} />
          <input
            placeholder={t("header.hospitalIdPlaceholder")}
            value={hospitalId}
            onChange={(e) => setHospitalId(e.target.value)}
            className="w-32 min-w-0 rounded bg-gray-200 px-2 py-1 text-xs text-gray-900 dark:bg-gray-800 dark:text-white sm:w-44 md:w-56"
          />
          <input
            placeholder={t("header.simulationIdPlaceholder")}
            value={simulationId}
            onChange={(e) => setSimulationId(e.target.value)}
            className="w-32 min-w-0 rounded bg-gray-200 px-2 py-1 text-xs text-gray-900 dark:bg-gray-800 dark:text-white sm:w-44 md:w-56"
          />
          <AuthBar />
          <LanguageSelector />
          <ThemeToggle />
        </div>
      </header>

      <main className="min-h-0 flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/twin" element={<TwinPage hospitalId={hospitalId} />} />
          <Route
            path="/simulator"
            element={<SimulatorPage hospitalId={hospitalId} onSimulationReady={setSimulationId} />}
          />
          <Route path="/reports" element={<ReportsPage simulationId={simulationId} />} />
          <Route path="/datasets" element={<PublicDatasetsPage />} />
          <Route path="/ml" element={<ModelComparisonPage />} />
          <Route path="/crisp" element={<MiningDashboardPage />} />
          <Route path="/crisp/projects" element={<CrispProjectsPage />} />
          <Route path="/crisp/:projectId" element={<CrispProjectDetailPage />} />
        </Routes>
      </main>
    </div>
  );
}
