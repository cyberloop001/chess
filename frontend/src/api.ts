/** API and WebSocket origins. Empty = same host (local Vite proxy). */

const STORAGE_KEY = "plyArenaApiUrl";

function bakedApiRoot(): string {
  return (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? "";
}

/** Runtime override (localStorage / ?api=) then VITE_API_URL. */
export function getApiRoot(): string {
  if (typeof window !== "undefined") {
    const fromQuery = new URLSearchParams(window.location.search).get("api");
    if (fromQuery) return fromQuery.replace(/\/$/, "");
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored) return stored.replace(/\/$/, "");
  }
  return bakedApiRoot();
}

export function setApiRoot(url: string): void {
  const clean = url.trim().replace(/\/$/, "");
  if (clean) window.localStorage.setItem(STORAGE_KEY, clean);
  else window.localStorage.removeItem(STORAGE_KEY);
}

export function apiUrl(path: string): string {
  const suffix = path.startsWith("/") ? path : `/${path}`;
  return `${getApiRoot()}${suffix}`;
}

export function wsUrl(path: string): string {
  const suffix = path.startsWith("/") ? path : `/${path}`;
  const explicit = (import.meta.env.VITE_WS_URL as string | undefined)?.replace(/\/$/, "");
  const root = getApiRoot();
  if (explicit) return `${explicit}${suffix}`;
  if (root) {
    const url = new URL(root);
    const proto = url.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${url.host}${suffix}`;
  }
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}${suffix}`;
}

export function assetUrl(path: string): string {
  const clean = path.replace(/^\//, "");
  return `${import.meta.env.BASE_URL}${clean}`;
}
