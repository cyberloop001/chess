/** API and WebSocket origins. Empty = same host (local Vite proxy). */

const API_ROOT = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? "";

export function apiUrl(path: string): string {
  const suffix = path.startsWith("/") ? path : `/${path}`;
  return `${API_ROOT}${suffix}`;
}

export function wsUrl(path: string): string {
  const suffix = path.startsWith("/") ? path : `/${path}`;
  const explicit = (import.meta.env.VITE_WS_URL as string | undefined)?.replace(/\/$/, "");
  if (explicit) return `${explicit}${suffix}`;
  if (API_ROOT) {
    const url = new URL(API_ROOT);
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
