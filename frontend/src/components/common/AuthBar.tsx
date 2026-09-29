import { useState } from "react";

import { useTranslation } from "@/i18n/useTranslation";
import { api } from "@/services/api";
import { useAuthStore } from "@/store/authStore";

/** Minimal login/register widget. There is no separate user-management UI in
 * this prototype: "Entrar / Registrar" logs in, and transparently registers
 * a new admin account first if the email doesn't exist yet (role "admin"
 * grants the "*" permission per app/api/v1/auth.py). */
export function AuthBar() {
  const { email, token, setAuth, logout } = useAuthStore();
  const { t } = useTranslation();
  const [formEmail, setFormEmail] = useState("admin@hospital-demo.example.com");
  const [password, setPassword] = useState("Admin123!");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loginOrRegister = async () => {
    setBusy(true);
    setError(null);
    try {
      let res;
      try {
        res = await api.post("/auth/login", { email: formEmail, password });
      } catch (loginErr) {
        try {
          await api.post("/auth/register", {
            email: formEmail,
            password,
            full_name: "Admin Demo",
            role_name: "admin",
          });
          res = await api.post("/auth/login", { email: formEmail, password });
        } catch {
          throw loginErr;
        }
      }
      setAuth(formEmail, res.data.access_token);
    } catch {
      setError(t("auth.error"));
    } finally {
      setBusy(false);
    }
  };

  if (token && email) {
    return (
      <div className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-300">
        <span className="rounded bg-emerald-100 px-2 py-1 text-emerald-700 ring-1 ring-emerald-300 dark:bg-emerald-950/50 dark:text-emerald-300 dark:ring-emerald-800/50">
          {email}
        </span>
        <button
          onClick={logout}
          className="rounded bg-gray-200 px-2 py-1 text-gray-700 hover:bg-gray-300 dark:bg-gray-800 dark:text-gray-200 dark:hover:bg-gray-700"
        >
          {t("auth.logout")}
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      <input
        value={formEmail}
        onChange={(e) => setFormEmail(e.target.value)}
        placeholder={t("auth.emailPlaceholder")}
        className="w-32 min-w-0 rounded bg-gray-200 px-2 py-1 text-xs text-gray-900 dark:bg-gray-800 dark:text-white sm:w-48 md:w-56"
      />
      <input
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        type="password"
        placeholder={t("auth.passwordPlaceholder")}
        className="w-24 min-w-0 rounded bg-gray-200 px-2 py-1 text-xs text-gray-900 dark:bg-gray-800 dark:text-white sm:w-32"
      />
      <button
        onClick={loginOrRegister}
        disabled={busy}
        className="whitespace-nowrap rounded bg-emerald-600 px-3 py-1 text-xs font-semibold text-white hover:bg-emerald-500 disabled:opacity-50"
      >
        {busy ? t("auth.loading") : t("auth.enterOrRegister")}
      </button>
      {error && <span className="text-xs text-red-600 dark:text-red-400">{error}</span>}
    </div>
  );
}
