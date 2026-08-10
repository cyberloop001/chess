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

type FinalResult = {
  result: string;
  winner: string;
  termination: string;
  white: string;
  black: string;
};

function formatResult(final: FinalResult): string {
  if (final.winner === "draw" || final.result === "1/2-1/2") {
    return `Draw (${final.result}) · ${final.termination}`;
  }
  if (final.winner === "mlp") {
    return `MLP + MCTS wins · ${final.result} · ${final.termination}`;
  }
  if (final.winner === "transformer") {
    return `Transformer + MCTS wins · ${final.result} · ${final.termination}`;
  }
  return `Result ${final.result} · ${final.termination}`;
}

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
  const [trainReports, setTrainReports] = useState<TrainModelReport[]>([]);
  const [finalResult, setFinalResult] = useState<FinalResult | null>(null);
  const matchMetaRef = useRef({ white: "mlp", black: "transformer", simulations: 64 });
  const movesRef = useRef<MoveEvent[]>([]);
  const pendingResultRef = useRef<FinalResult | null>(null);
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

  function persistMatch(end: FinalResult) {
    const stored = buildStoredMatch({
      white: end.white,
      black: end.black,
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
        setFinalResult(null);
        pendingResultRef.current = null;
        setStatus(`${modelLabel(event.white)} vs ${modelLabel(event.black)}`);
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
        setFinalResult(null);
        pendingResultRef.current = null;
        setRunningState(true);
        setStatus(`${modelLabel(event.white)} (White) vs ${modelLabel(event.black)} (Black)`);
        break;
      case "thinking":
        setThinkingSide(event.side);
        setStatus(`${modelLabel(event.model)} is searching…`);
        break;
      case "move": {
        movesRef.current = [...movesRef.current, event];
        setFen(event.fen);
        setMoves((prev) => [...prev, event]);
        setLastUci(event.uci);
        setLatestMcts(event.mcts);
        setThinkingSide(null);
        setStatus(`${modelLabel(event.model)} played ${event.san}`);
        break;
      }
      case "match_end": {
        setThinkingSide(null);
        setFen(event.fen);
        const end: FinalResult = {
          result: event.result,
          winner: event.winner,
          termination: event.termination,
          white: event.white || matchMetaRef.current.white,
          black: event.black || matchMetaRef.current.black,
        };
        pendingResultRef.current = end;
        setFinalResult(end);
        setStatus(formatResult(end));
        persistMatch(end);
        break;
      }
      case "training_complete": {
        setTrainReports(event.models);
        const summary = event.models
          .map((m) => `${modelLabel(m.model)} loss ${m.total_loss}`)
          .join(" · ");
        const base = pendingResultRef.current
          ? formatResult(pendingResultRef.current)
          : "Game finished";
        setStatus(`${base} · trained (${summary})`);
        break;
      }
      case "series_end": {
        setRunningState(false);
        setThinkingSide(null);
        const end: FinalResult = pendingResultRef.current ?? {
          result: event.result || "unknown",
          winner: event.series_winner,
          termination: event.termination || "unknown",
          white: event.white || matchMetaRef.current.white,
          black: event.black || matchMetaRef.current.black,
        };
        setFinalResult(end);
        setStatus(formatResult(end));
        break;
      }
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
    setFinalResult(null);
    pendingResultRef.current = null;
    setStatus("Connecting…");
    const ws = ensureSocket();
    const payload = {
      action: "start",
      white_model: "mlp",
      black_model: "transformer",
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

  const resultKind =
    finalResult == null
      ? null
      : finalResult.winner === "draw" || finalResult.result === "1/2-1/2"
        ? "draw"
        : finalResult.winner === "mlp"
          ? "mlp"
          : "transformer";

  return (
    <main className="layout">
      <section className="panel">
        <h2>Match setup</h2>
        <p className="setup-note">
          Fixed duel: White = MLP + MCTS, Black = Transformer + MCTS. Plays one game, shows the
          result (win / loss / draw), then both models self-train.
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
            Start duel
          </button>
          <button className="btn btn-ghost" disabled={!running} onClick={cancelMatch}>
            Cancel
          </button>
        </div>
        <div className="status-pill">
          <span className={`dot ${running ? "" : "idle"}`} />
          {status}
        </div>
        {finalResult && (
          <div className={`result-banner result-${resultKind}`}>
            <span className="result-label">Final result</span>
            <strong>{formatResult(finalResult)}</strong>
          </div>
        )}
        {trainReports.length > 0 && (
          <div className="train-box">
            <strong>Training after game</strong>
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
            <span>White</span>
            <strong>{modelLabel(whiteModel)}</strong>
          </div>
          <div className={`player ${thinkingSide === "black" ? "active" : ""}`}>
            <span>Black</span>
            <strong>{modelLabel(blackModel)}</strong>
          </div>
        </div>
        <ChessBoard fen={fen} lastUci={lastUci} />
        {finalResult && (
          <div className={`result-banner result-banner-board result-${resultKind}`}>
            <strong>{formatResult(finalResult)}</strong>
          </div>
        )}
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
