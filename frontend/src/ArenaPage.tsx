import { useEffect, useMemo, useRef, useState } from "react";
import { ChessBoard } from "./ChessBoard";
import { buildStoredMatch, loadMatchHistory, saveMatchHistory } from "./matchHistory";
import {
  modelLabel,
  type MatchEvent,
  type ModelId,
  type MoveEvent,
  type TrainModelReport,
} from "./types";

const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

type Props = {
  onHistoryChange?: () => void;
  onRunningChange?: (running: boolean) => void;
};

export function ArenaPage({ onHistoryChange, onRunningChange }: Props) {
  const [whiteModel, setWhiteModel] = useState<ModelId>("mlp");
  const [blackModel, setBlackModel] = useState<ModelId>("transformer");
  const [fen, setFen] = useState(START_FEN);
  const [moves, setMoves] = useState<MoveEvent[]>([]);
  const [status, setStatus] = useState("Ready for a duel");
  const [running, setRunning] = useState(false);
  const [thinkingSide, setThinkingSide] = useState<"white" | "black" | null>(null);
  const [lastUci, setLastUci] = useState<string | null>(null);
  const [latestMcts, setLatestMcts] = useState<MoveEvent["mcts"] | null>(null);
  const [gameIndex, setGameIndex] = useState(0);
  const [trainReports, setTrainReports] = useState<TrainModelReport[]>([]);
  const matchMetaRef = useRef({ white: "mlp", black: "transformer", simulations: 64 });
  const movesRef = useRef<MoveEvent[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const onRunningChangeRef = useRef(onRunningChange);
  onRunningChangeRef.current = onRunningChange;

  const wsUrl = useMemo(() => {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${window.location.host}/ws/match`;
  }, []);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  function setRunningState(next: boolean) {
    setRunning(next);
    onRunningChangeRef.current?.(next);
  }

  function persistMatch(end: {
    result: string;
    winner: string;
    termination: string;
    white?: string;
    black?: string;
  }) {
    const stored = buildStoredMatch({
      white: end.white || matchMetaRef.current.white,
      black: end.black || matchMetaRef.current.black,
      simulations: matchMetaRef.current.simulations,
      result: end.result,
      winner: end.winner,
      termination: end.termination,
      moves: movesRef.current,
    });
    const next = [stored, ...loadMatchHistory()];
    saveMatchHistory(next);
    onHistoryChange?.();
  }

  function ensureSocket(): WebSocket {
    if (wsRef.current && wsRef.current.readyState <= WebSocket.OPEN) {
      return wsRef.current;
    }
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    ws.onmessage = (msg) => {
      handleEvent(JSON.parse(msg.data) as MatchEvent);
    };
    ws.onclose = () => {
      setRunningState(false);
      setThinkingSide(null);
    };
    return ws;
  }

  function handleEvent(event: MatchEvent) {
    switch (event.type) {
      case "series_start":
        setRunningState(true);
        setTrainReports([]);
        setStatus(
          `Series started · play until one wins · ${modelLabel(event.white)} vs ${modelLabel(event.black)}`,
        );
        break;
      case "match_start":
        matchMetaRef.current = {
          white: event.white,
          black: event.black,
          simulations: event.simulations,
        };
        movesRef.current = [];
        setWhiteModel(event.white as ModelId);
        setBlackModel(event.black as ModelId);
        setFen(event.fen);
        setMoves([]);
        setLastUci(null);
        setLatestMcts(null);
        setGameIndex(event.game_index ?? 1);
        setRunningState(true);
        setStatus(
          `Game ${event.game_index ?? 1} · ${modelLabel(event.white)} (W) vs ${modelLabel(event.black)} (B)`,
        );
        break;
      case "thinking":
        setThinkingSide(event.side);
        setStatus(
          `Game ${(event.game_index ?? gameIndex) || 1} · ${modelLabel(event.model)} is searching…`,
        );
        break;
      case "move": {
        movesRef.current = [...movesRef.current, event];
        setFen(event.fen);
        setMoves((prev) => [...prev, event]);
        setLastUci(event.uci);
        setLatestMcts(event.mcts);
        setThinkingSide(null);
        setStatus(
          `Game ${(event.game_index ?? gameIndex) || 1} · ${modelLabel(event.model)} played ${event.san}`,
        );
        break;
      }
      case "match_end":
        setThinkingSide(null);
        setFen(event.fen);
        if (event.winner === "draw") {
          setStatus(`Game ${event.game_index ?? gameIndex} drawn (${event.result}) · rematching…`);
        } else {
          setStatus(
            `Game ${event.game_index ?? gameIndex} · ${modelLabel(event.winner)} wins (${event.result})`,
          );
        }
        persistMatch({
          result: event.result,
          winner: event.winner,
          termination: event.termination,
          white: event.white,
          black: event.black,
        });
        break;
      case "training_complete": {
        setTrainReports(event.models);
        const summary = event.models
          .map((m) => `${modelLabel(m.model)} loss ${m.total_loss}`)
          .join(" · ");
        setStatus(`Game ${event.game} trained · ${summary}`);
        break;
      }
      case "series_end":
        setRunningState(false);
        setThinkingSide(null);
        if (event.series_winner === "none") {
          setStatus(`Series over · ${event.game_count} games · no decisive winner`);
        } else {
          setStatus(
            `Series over · ${modelLabel(event.series_winner)} wins after ${event.game_count} game(s)`,
          );
        }
        break;
      case "match_cancelled":
        setRunningState(false);
        setThinkingSide(null);
        setStatus("Match cancelled");
        break;
      case "error":
        setRunningState(false);
        setStatus(event.message);
        break;
      default:
        break;
    }
  }

  function startMatch() {
    setFen(START_FEN);
    setMoves([]);
    movesRef.current = [];
    setLastUci(null);
    setLatestMcts(null);
    setTrainReports([]);
    setGameIndex(0);
    setStatus("Connecting…");
    const ws = ensureSocket();
    const payload = {
      action: "start",
      white_model: "mlp",
      black_model: "transformer",
      play_until_win: true,
      max_games: 8,
    };
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
      setRunningState(true);
    } else {
      ws.onopen = () => {
        ws.send(JSON.stringify(payload));
        setRunningState(true);
      };
    }
  }

  function cancelMatch() {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "cancel" }));
    }
  }

  return (
    <main className="layout">
      <section className="panel">
        <h2>Match setup</h2>
        <p className="setup-note">
          Fixed duel: White = MLP + MCTS, Black = Transformer + MCTS. Keeps playing until one model
          wins, then both self-train.
        </p>
        <div className="pairing">
          <div>
            <span className="pairing-label">White</span>
            <strong>{modelLabel(whiteModel)}</strong>
          </div>
          <div>
            <span className="pairing-label">Black</span>
            <strong>{modelLabel(blackModel)}</strong>
          </div>
        </div>
        <div className="actions">
          <button className="btn btn-primary" disabled={running} onClick={startMatch}>
            Start series
          </button>
          <button className="btn btn-ghost" disabled={!running} onClick={cancelMatch}>
            Cancel
          </button>
        </div>
        <div className="status-pill">
          <span className={`dot ${running ? "" : "idle"}`} />
          {status}
        </div>
        {trainReports.length > 0 && (
          <div className="train-box">
            <strong>Last training</strong>
            <ul>
              {trainReports.map((r) => (
                <li key={r.model}>
                  {modelLabel(r.model)} · {r.samples} samples · loss {r.total_loss}
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>

      <section className="board-wrap">
        <div className="players">
          <div className={`player ${thinkingSide === "white" ? "active" : ""}`}>
            <span>White{gameIndex ? ` · G${gameIndex}` : ""}</span>
            <strong>{modelLabel(whiteModel)}</strong>
          </div>
          <div className={`player ${thinkingSide === "black" ? "active" : ""}`}>
            <span>Black{gameIndex ? ` · G${gameIndex}` : ""}</span>
            <strong>{modelLabel(blackModel)}</strong>
          </div>
        </div>
        <ChessBoard fen={fen} lastUci={lastUci} />
      </section>

      <section className="panel">
        <h2>Moves</h2>
        <ol className="move-list">
          {moves.map((m) => (
            <li key={`${m.game_index ?? 0}-${m.ply}`}>
              <span>{m.ply}.</span>
              <span className="san">{m.san}</span>
              <span className="model">{modelLabel(m.model)}</span>
            </li>
          ))}
        </ol>
        {latestMcts && (
          <div className="mcts-box">
            <div>
              Root value {latestMcts.root_value} · {latestMcts.simulations} sims
            </div>
            <ol>
              {latestMcts.top_moves.map((tm) => (
                <li key={tm.uci}>
                  {tm.uci} · v{tm.visits} · q {tm.q}
                </li>
              ))}
            </ol>
          </div>
        )}
      </section>
    </main>
  );
}
