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
  game_index?: number;
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
  white?: string;
  black?: string;
  game_index?: number;
};

export type TrainModelReport = {
  model: string;
  steps: number;
  policy_loss: number;
  value_loss: number;
  total_loss: number;
  samples: number;
  saved_to: string | null;
};

export type MatchEvent =
  | {
      type: "series_start";
      white: string;
      black: string;
      simulations: number;
      play_until_win: boolean;
      max_games: number;
    }
  | {
      type: "match_start";
      fen: string;
      white: string;
      black: string;
      simulations: number;
      game_index?: number;
    }
  | {
      type: "thinking";
      side: "white" | "black";
      model: string;
      fen: string;
      ply: number;
      game_index?: number;
    }
  | MoveEvent
  | MatchEndEvent
  | {
      type: "training_complete";
      game: number;
      models: TrainModelReport[];
    }
  | {
      type: "series_end";
      game_count: number;
      series_winner: string;
      result?: string;
      termination?: string;
      white?: string;
      black?: string;
      games: Array<{ result: string; winner: string; white: string; black: string }>;
    }
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
  if (id === "none") return "None";
  return id;
}

export function shortModel(id: string): string {
  if (id === "mlp") return "MLP";
  if (id === "transformer") return "Transformer";
  return id;
}
