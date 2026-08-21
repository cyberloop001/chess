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
  mcts?: {
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
  mode?: string;
  model?: string;
  human_color?: string;
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
      train_count?: number;
    }
  | {
      type: "match_start";
      fen: string;
      white: string;
      black: string;
      simulations: number;
      game_index?: number;
      mode?: string;
      human_color?: "white" | "black";
      model?: string;
    }
  | {
      type: "your_turn";
      side: "white" | "black";
      fen: string;
      ply: number;
      legal_moves: string[];
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
      type: "training_start";
      game?: number;
      train_count?: number;
      model?: string;
    }
  | {
      type: "training_complete";
      game?: number;
      train_count?: number;
      model?: string;
      models: TrainModelReport[];
    }
  | {
      type: "series_end";
      game_count: number;
      train_count?: number;
      series_winner: string;
      result?: string;
      termination?: string;
      white?: string;
      black?: string;
      games?: Array<{ result: string; winner: string; white: string; black: string }>;
      winner?: string;
      fen?: string;
      mode?: string;
      model?: string;
      human_color?: string;
    }
  | { type: "analytics_saved"; match_id: string }
  | { type: "match_cancelled"; fen: string }
  | { type: "error"; message: string }
  | { type: "info"; message: string }
  | { type: "pong" };

export type SelfPlayRecord = {
  id: string;
  runId: string;
  playedAt: string;
  model: string;
  game: number;
  gamesInRun: number;
  simulations: number;
  plies: number;
  result: string;
  termination: string;
  policyLoss: number;
  valueLoss: number;
  totalLoss: number;
  samples: number;
  replaySize: number;
  savedTo: string | null;
};

export type SelfPlayEvent =
  | {
      type: "selfplay_start";
      run_id: string;
      model: string;
      games: number;
      simulations: number;
    }
  | {
      type: "selfplay_thinking";
      run_id: string;
      model: string;
      game: number;
      games: number;
      fen?: string;
    }
  | {
      type: "selfplay_move";
      run_id: string;
      model: string;
      game: number;
      games: number;
      ply: number;
      side: "white" | "black";
      uci: string;
      san: string;
      fen: string;
      root_value: number;
      net_value?: number;
      move_q?: number;
      simulations: number;
    }
  | {
      type: "selfplay_training";
      run_id: string;
      model: string;
      game: number;
      games: number;
      plies: number;
      result: string;
      termination: string;
      fen?: string;
    }
  | {
      type: "selfplay_train_step";
      run_id: string;
      model: string;
      game: number;
      games: number;
      step: number;
      epoch: number;
      epochs: number;
      policy_loss: number;
      value_loss: number;
      total_loss: number;
      grad_norm: number;
      samples: number;
    }
  | {
      type: "selfplay_weights";
      run_id: string;
      model: string;
      game: number;
      games: number;
      steps: number;
      samples: number;
      lr: number;
      epochs: number;
      layer_deltas: Array<{
        name: string;
        short: string;
        numel: number;
        mean_abs_delta: number;
        rms_delta: number;
        rel_mean: number;
      }>;
      weight_hist: {
        counts: number[];
        edges: number[];
        mean_rel: number;
        median_rel: number;
        sampled: number;
      } | null;
      train_trace: Array<{
        step: number;
        epoch: number;
        policy_loss: number;
        value_loss: number;
        total_loss: number;
        grad_norm: number;
      }>;
    }
  | (SelfPlayRecord & { type: "selfplay_game"; games: number })
  | {
      type: "selfplay_complete" | "selfplay_cancelled";
      run_id: string;
      model: string;
      game_count: number;
      games: number;
    }
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
  if (id === "human") return "Human";
  if (id === "none") return "None";
  return id;
}

export function shortModel(id: string): string {
  if (id === "mlp") return "MLP";
  if (id === "transformer") return "Transformer";
  if (id === "human") return "Human";
  return id;
}
