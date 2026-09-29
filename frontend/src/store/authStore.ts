import { create } from "zustand";

interface AuthState {
  email: string | null;
  token: string | null;
  setAuth: (email: string, token: string) => void;
  logout: () => void;
}

function readStored(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

export const useAuthStore = create<AuthState>((set) => ({
  email: readStored("user_email"),
  token: readStored("access_token"),
  setAuth: (email, token) => {
    try {
      localStorage.setItem("access_token", token);
      localStorage.setItem("user_email", email);
    } catch {
      // localStorage unavailable (private mode, etc.) — auth still works for this tab via state
    }
    set({ email, token });
  },
  logout: () => {
    try {
      localStorage.removeItem("access_token");
      localStorage.removeItem("user_email");
    } catch {
      // ignore
    }
    set({ email: null, token: null });
  },
}));
