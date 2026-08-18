import type { SelfPlayRecord } from "./types";

export async function fetchSelfPlayHistory(): Promise<SelfPlayRecord[]> {
  const res = await fetch("/api/analytics/selfplay");
  if (!res.ok) {
    throw new Error(`Failed to load self-play history (${res.status})`);
  }
  const data = (await res.json()) as { games?: SelfPlayRecord[] };
  return Array.isArray(data.games) ? data.games : [];
}

export async function clearSelfPlayHistoryRemote(): Promise<void> {
  const res = await fetch("/api/analytics/selfplay", { method: "DELETE" });
  if (!res.ok) {
    throw new Error(`Failed to clear self-play history (${res.status})`);
  }
}

export type ModelImprovement = {
  id: string;
  games: number;
  earlyLoss: number;
  lateLoss: number;
  lossDelta: number;
  lossDropPct: number;
  earlyPlies: number;
  latePlies: number;
  plyDelta: number;
  earlyDecisive: number;
  lateDecisive: number;
  decisiveDelta: number;
};

function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function isDecisive(g: SelfPlayRecord): boolean {
  const t = (g.termination || "").toLowerCase();
  return t.includes("checkmate") || t.includes("resign") || g.result === "1-0" || g.result === "0-1";
}

function chrono(games: SelfPlayRecord[]): SelfPlayRecord[] {
  return [...games].sort((a, b) => a.playedAt.localeCompare(b.playedAt));
}

export function improvementFor(games: SelfPlayRecord[], model: string): ModelImprovement | null {
  const series = chrono(games.filter((g) => g.model === model));
  if (series.length < 2) return null;
  const n = Math.max(1, Math.ceil(series.length * 0.25));
  const early = series.slice(0, n);
  const late = series.slice(-n);
  const earlyLoss = mean(early.map((g) => g.totalLoss));
  const lateLoss = mean(late.map((g) => g.totalLoss));
  const earlyPlies = mean(early.map((g) => g.plies));
  const latePlies = mean(late.map((g) => g.plies));
  const earlyDecisive = early.filter(isDecisive).length / early.length;
  const lateDecisive = late.filter(isDecisive).length / late.length;
  const lossDelta = earlyLoss - lateLoss;
  return {
    id: model,
    games: series.length,
    earlyLoss,
    lateLoss,
    lossDelta,
    lossDropPct: earlyLoss > 1e-6 ? lossDelta / earlyLoss : 0,
    earlyPlies,
    latePlies,
    plyDelta: earlyPlies - latePlies,
    earlyDecisive,
    lateDecisive,
    decisiveDelta: lateDecisive - earlyDecisive,
  };
}

export const SELFPLAY_CITATIONS = [
  {
    id: "az-arxiv",
    title: "Silver et al., 2017. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm.",
    href: "https://arxiv.org/abs/1712.01815",
    note: "Policy target = MCTS visit distribution π; value target = game outcome z ∈ {−1, 0, +1}.",
  },
  {
    id: "az-science",
    title: "Silver et al., 2018. A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. Science.",
    href: "https://www.science.org/doi/10.1126/science.aar6404",
    note: "Same loop: self-play → train policy/value → stronger search. Loss drop is the training signal, not Elo.",
  },
] as const;
