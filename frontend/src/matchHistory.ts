import { apiUrl } from "./api";
import type { MoveEvent, StoredMatch } from "./types";

export async function fetchMatchHistory(): Promise<StoredMatch[]> {
  const res = await fetch(apiUrl("/api/analytics/matches"));
  if (!res.ok) {
    throw new Error(`Failed to load analytics (${res.status})`);
  }
  const data = (await res.json()) as { matches?: StoredMatch[] };
  return Array.isArray(data.matches) ? data.matches : [];
}

export async function clearMatchHistoryRemote(): Promise<void> {
  const res = await fetch(apiUrl("/api/analytics/matches"), { method: "DELETE" });
  if (!res.ok) {
    throw new Error(`Failed to clear analytics (${res.status})`);
  }
}

export function buildStoredMatch(input: {
  white: string;
  black: string;
  simulations: number;
  result: string;
  winner: string;
  termination: string;
  moves: MoveEvent[];
}): StoredMatch {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    playedAt: new Date().toISOString(),
    white: input.white,
    black: input.black,
    simulations: input.simulations,
    result: input.result,
    winner: input.winner,
    termination: input.termination,
    moves: input.moves.map((m) => {
      const visits = (m.mcts?.top_moves ?? []).map((t) => t.visits);
      const topVisits = visits[0] ?? 0;
      const totalTopVisits = visits.reduce((a, b) => a + b, 0) || 1;
      return {
        ply: m.ply,
        side: m.side,
        model: m.model,
        uci: m.uci,
        san: m.san,
        rootValue: m.mcts?.root_value ?? 0,
        topVisits,
        totalTopVisits,
      };
    }),
  };
}

export type ModelStats = {
  id: string;
  games: number;
  wins: number;
  losses: number;
  draws: number;
  winRate: number;
  avgPly: number;
  avgRootValue: number;
};

export type AnalyticsSummary = {
  totalMatches: number;
  finishedMatches: number;
  avgPlies: number;
  mlp: ModelStats;
  transformer: ModelStats;
  headToHead: {
    mlpWins: number;
    transformerWins: number;
    draws: number;
  };
  recent: StoredMatch[];
};

function emptyStats(id: string): ModelStats {
  return {
    id,
    games: 0,
    wins: 0,
    losses: 0,
    draws: 0,
    winRate: 0,
    avgPly: 0,
    avgRootValue: 0,
  };
}

function normalizeModel(id: string): "mlp" | "transformer" | "other" {
  const key = id.toLowerCase();
  if (key.includes("mlp")) return "mlp";
  if (key.includes("transformer")) return "transformer";
  return "other";
}

export function computeAnalytics(matches: StoredMatch[]): AnalyticsSummary {
  const mlp = emptyStats("mlp");
  const transformer = emptyStats("transformer");
  const headToHead = { mlpWins: 0, transformerWins: 0, draws: 0 };
  let plySum = 0;
  let finished = 0;

  const valueAcc: Record<string, { sum: number; n: number; plySum: number }> = {
    mlp: { sum: 0, n: 0, plySum: 0 },
    transformer: { sum: 0, n: 0, plySum: 0 },
  };

  for (const match of matches) {
    if (!match.result || match.result === "cancelled") continue;
    finished += 1;
    plySum += match.moves.length;

    const white = normalizeModel(String(match.white));
    const black = normalizeModel(String(match.black));
    const winner = normalizeModel(String(match.winner));

    for (const side of [white, black]) {
      if (side === "mlp") mlp.games += 1;
      if (side === "transformer") transformer.games += 1;
    }

    if (match.result === "1/2-1/2") {
      if (white === "mlp" || black === "mlp") mlp.draws += 1;
      if (white === "transformer" || black === "transformer") transformer.draws += 1;
      if (
        (white === "mlp" && black === "transformer") ||
        (white === "transformer" && black === "mlp")
      ) {
        headToHead.draws += 1;
      }
    } else if (winner === "mlp") {
      mlp.wins += 1;
      if (black === "transformer" || white === "transformer") transformer.losses += 1;
      if (
        (white === "mlp" && black === "transformer") ||
        (white === "transformer" && black === "mlp")
      ) {
        headToHead.mlpWins += 1;
      }
    } else if (winner === "transformer") {
      transformer.wins += 1;
      if (black === "mlp" || white === "mlp") mlp.losses += 1;
      if (
        (white === "mlp" && black === "transformer") ||
        (white === "transformer" && black === "mlp")
      ) {
        headToHead.transformerWins += 1;
      }
    } else {
      // Human (or other) won — count a loss for the model that played
      if (white === "mlp" || black === "mlp") mlp.losses += 1;
      if (white === "transformer" || black === "transformer") transformer.losses += 1;
    }

    for (const move of match.moves) {
      const model = normalizeModel(move.model);
      if (model === "other") continue;
      valueAcc[model].sum += move.rootValue;
      valueAcc[model].n += 1;
    }
    if (white === "mlp" || black === "mlp") valueAcc.mlp.plySum += match.moves.length;
    if (white === "transformer" || black === "transformer") {
      valueAcc.transformer.plySum += match.moves.length;
    }
  }

  for (const stats of [mlp, transformer]) {
    const played = stats.wins + stats.losses + stats.draws;
    stats.winRate = played ? stats.wins / played : 0;
    stats.avgRootValue = valueAcc[stats.id].n ? valueAcc[stats.id].sum / valueAcc[stats.id].n : 0;
    stats.avgPly = stats.games ? valueAcc[stats.id].plySum / stats.games : 0;
  }

  return {
    totalMatches: matches.length,
    finishedMatches: finished,
    avgPlies: finished ? plySum / finished : 0,
    mlp,
    transformer,
    headToHead,
    recent: matches.slice(0, 12),
  };
}
