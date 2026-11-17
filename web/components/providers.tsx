"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import { api } from "@/lib/api";
import { readStorage, useAsync, useInterval, writeStorage } from "@/lib/hooks";
import type { HealthResponse, UserResponse } from "@/lib/types";

// ---- theme ---------------------------------------------------------------------------------
type Theme = "light" | "dark";

interface ThemeCtx {
  theme: Theme;
  toggle: () => void;
}

const ThemeContext = createContext<ThemeCtx>({ theme: "light", toggle: () => {} });

function useThemeState(): ThemeCtx {
  const [theme, setTheme] = useState<Theme>("light");
  useEffect(() => {
    setTheme(document.documentElement.classList.contains("dark") ? "dark" : "light");
  }, []);
  const toggle = useCallback(() => {
    setTheme((prev) => {
      const next: Theme = prev === "dark" ? "light" : "dark";
      document.documentElement.classList.toggle("dark", next === "dark");
      writeStorage("ragops-theme", next);
      return next;
    });
  }, []);
  return { theme, toggle };
}

export const useTheme = () => useContext(ThemeContext);

// ---- persona + health ------------------------------------------------------------------------
interface AppCtx {
  users: UserResponse[];
  usersError: unknown;
  userId: string;
  user: UserResponse | undefined;
  setUserId: (id: string) => void;
  health: HealthResponse | undefined;
  healthError: unknown;
  refreshHealth: () => Promise<void>;
}

const AppContext = createContext<AppCtx | null>(null);

export function useApp(): AppCtx {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used inside <Providers>");
  return ctx;
}

const PERSONA_KEY = "ragops-persona";
const DEFAULT_PERSONA = "priya.sharma";

export function Providers({ children }: { children: React.ReactNode }) {
  const themeState = useThemeState();
  const users = useAsync(() => api.users(), []);
  const health = useAsync(() => api.health(), []);
  const [userId, setUserIdState] = useState<string>("");

  // Poll health so the sidebar dot reflects the API coming up / going down.
  useInterval(() => void health.reload(), 15000, true);

  useEffect(() => {
    if (!users.data?.length || userId) return;
    const stored = readStorage(PERSONA_KEY);
    const pick =
      users.data.find((u) => u.user_id === stored) ??
      users.data.find((u) => u.user_id === DEFAULT_PERSONA) ??
      users.data[0];
    setUserIdState(pick.user_id);
  }, [users.data, userId]);

  const setUserId = useCallback((id: string) => {
    setUserIdState(id);
    writeStorage(PERSONA_KEY, id);
  }, []);

  const value = useMemo<AppCtx>(
    () => ({
      users: users.data ?? [],
      usersError: users.error,
      userId,
      user: users.data?.find((u) => u.user_id === userId),
      setUserId,
      health: health.data,
      healthError: health.error,
      refreshHealth: health.reload,
    }),
    [users.data, users.error, userId, setUserId, health.data, health.error, health.reload],
  );

  return (
    <ThemeContext.Provider value={themeState}>
      <AppContext.Provider value={value}>{children}</AppContext.Provider>
    </ThemeContext.Provider>
  );
}
