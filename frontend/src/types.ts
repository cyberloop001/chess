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
  | {
      type: "match_end";
      result: string;
      fen: string;
      winner: string;
      termination: string;
    }
  | { type: "match_cancelled"; fen: string }
  | { type: "error"; message: string }
  | { type: "info"; message: string }
  | { type: "pong" };

export function modelLabel(id: string): string {
  if (id === "mlp") return "MLP + MCTS";
  if (id === "transformer") return "Transformer + MCTS";
  return id;
}
