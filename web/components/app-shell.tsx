"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import {
  BarChart3,
  Database,
  FlaskConical,
  GitCompare,
  LayoutDashboard,
  Menu,
  MessageSquareText,
  Moon,
  ShieldCheck,
  Sun,
  X,
} from "lucide-react";

import { useApp, useTheme } from "@/components/providers";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/chat", label: "RAG Chat", icon: MessageSquareText },
  { href: "/documents", label: "Knowledge Base", icon: Database },
  { href: "/evals", label: "Evaluations", icon: BarChart3 },
  { href: "/evals/compare", label: "Compare Runs", icon: GitCompare },
  { href: "/sandbox", label: "Prompt Sandbox", icon: FlaskConical },
];

function isActive(pathname: string, href: string) {
  if (href === "/") return pathname === "/";
  if (href === "/evals") return pathname === "/evals" || (pathname.startsWith("/evals/") && !pathname.startsWith("/evals/compare"));
  return pathname === href || pathname.startsWith(`${href}/`);
}

function HealthDot() {
  const { health, healthError } = useApp();
  const online = Boolean(health) && !healthError;
  const label = !online
    ? "API offline"
    : health?.offline
      ? "API online (offline mode)"
      : "API online";
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground" title={label}>
      <span className="relative flex h-2.5 w-2.5">
        {online && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-success/60 opacity-60" />}
        <span className={cn("relative inline-flex h-2.5 w-2.5 rounded-full", online ? "bg-success" : "bg-destructive")} />
      </span>
      <span>{label}</span>
      {online && health?.version && <span className="ml-auto font-mono">v{health.version}</span>}
    </div>
  );
}

function PersonaSelect() {
  const { users, userId, setUserId, user, usersError } = useApp();
  return (
    <div className="space-y-2">
      <Label htmlFor="persona">Acting as (ACL persona)</Label>
      <Select
        id="persona"
        value={userId}
        onChange={(e) => setUserId(e.target.value)}
        disabled={!users.length}
        aria-describedby="persona-groups"
      >
        {!users.length && <option value="">{usersError ? "Users unavailable" : "Loading..."}</option>}
        {users.map((u) => (
          <option key={u.user_id} value={u.user_id}>
            {u.name} - {u.title}
          </option>
        ))}
      </Select>
      {user && (
        <div id="persona-groups" className="flex flex-wrap gap-1">
          {["public", ...user.groups.filter((g) => g !== "public")].map((g) => (
            <Badge key={g} variant="muted" className="font-mono text-[10px]">
              {g}
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}

function ThemeToggle() {
  const { theme, toggle } = useTheme();
  return (
    <Button variant="ghost" size="icon" onClick={toggle} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>
      {theme === "dark" ? <Sun /> : <Moon />}
    </Button>
  );
}

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();
  return (
    <div className="flex h-full flex-col gap-6 p-4">
      <div className="flex items-center gap-2 px-1">
        <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary text-primary-foreground">
          <ShieldCheck className="h-4 w-4" />
        </div>
        <div className="leading-tight">
          <div className="text-sm font-semibold">RAG-Ops</div>
          <div className="text-[11px] text-muted-foreground">Knowledge Engine Console</div>
        </div>
      </div>
      <nav aria-label="Main" className="flex flex-col gap-0.5">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = isActive(pathname, href);
          return (
            <Link
              key={href}
              href={href}
              onClick={onNavigate}
              aria-current={active ? "page" : undefined}
              className={cn(
                "flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm transition-colors",
                active ? "bg-accent font-medium text-foreground" : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
              )}
            >
              <Icon className="h-4 w-4" />
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="mt-auto space-y-4 border-t pt-4">
        <PersonaSelect />
        <div className="flex items-center justify-between gap-2">
          <HealthDot />
          <ThemeToggle />
        </div>
      </div>
    </div>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();
  useEffect(() => setOpen(false), [pathname]);

  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-64 shrink-0 border-r bg-card/50 lg:block">
        <SidebarContent />
      </aside>

      {open && (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
          <button className="absolute inset-0 bg-background/70 backdrop-blur-sm" aria-label="Close navigation" onClick={() => setOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-72 max-w-[85vw] border-r bg-card shadow-xl">
            <Button variant="ghost" size="icon" className="absolute right-2 top-3" onClick={() => setOpen(false)} aria-label="Close navigation">
              <X />
            </Button>
            <SidebarContent onNavigate={() => setOpen(false)} />
          </aside>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b bg-background/90 px-4 backdrop-blur lg:hidden">
          <Button variant="ghost" size="icon" onClick={() => setOpen(true)} aria-label="Open navigation">
            <Menu />
          </Button>
          <span className="text-sm font-semibold">RAG-Ops</span>
          <div className="ml-auto">
            <HealthDot />
          </div>
        </header>
        <main className="min-w-0 flex-1 px-4 py-6 sm:px-6 lg:px-8">{children}</main>
      </div>
    </div>
  );
}
