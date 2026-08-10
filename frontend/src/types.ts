export type ModelId = "mlp" | "transformer";

export type MctsMove = {
  uci: string;
  visits: number;
  prior: number;
  q: number;
};

export type MoveEvent = {
  type: "move";
  ply: number;
  side: "white" | "black";
  model: string;
  uci: string;
  san: string;
  fen: string;
  mcts: {
    simulations: number;
    root_value: number;
    top_moves: MctsMove[];
  };
};

export type MatchEndEvent = {
  type: "match_end";
  result: string;
  fen: string;
  winner: string;
  termination: string;
};

export type MatchEvent =
  | {
      type: "match_start";
      fen: string;
      white: string;
      black: string;
      simulations: number;
    }
  | {
      type: "thinking";
      side: "white" | "black";
      model: string;
      fen: string;
      ply: number;
    }
  | MoveEvent
  | MatchEndEvent
  | { type: "match_cancelled"; fen: string }
  | { type: "error"; message: string }
  | { type: "info"; message: string }
  | { type: "pong" };

export type StoredMatch = {
  id: string;
  playedAt: string;
  white: ModelId | string;
  black: ModelId | string;
  simulations: number;
  result: string;
  winner: string;
  termination: string;
  moves: Array<{
    ply: number;
    side: "white" | "black";
    model: string;
    uci: string;
    san: string;
    rootValue: number;
    topVisits: number;
    totalTopVisits: number;
  }>;
};

export function modelLabel(id: string): string {
  if (id === "mlp") return "MLP + MCTS";
  if (id === "transformer") return "Transformer + MCTS";
  return id;
}

export function shortModel(id: string): string {
  if (id === "mlp") return "MLP";
  if (id === "transformer") return "Transformer";
  return id;
}
