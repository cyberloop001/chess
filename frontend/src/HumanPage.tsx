import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Chess } from "chess.js";
import { wsUrl as humanSocketUrl } from "./api";
import { ChessBoard } from "./ChessBoard";
import { explainSan } from "./moveExplain";
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

function formatTermination(raw: string): string {
  return raw
    .toLowerCase()
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

function formatResult(final: FinalResult): string {
  const why = formatTermination(final.termination);
  if (final.winner === "draw" || final.result === "1/2-1/2") {
    return `Draw · ${why}`;
  }
  if (final.winner === "human") {
    return `You win · ${final.result} · ${why}`;
  }
  return `${modelLabel(final.winner)} wins · ${final.result} · ${why}`;
}

export function HumanPage({ onHistoryChange, onRunningChange }: Props) {
  const [humanColor, setHumanColor] = useState<"white" | "black">("white");
  const [opponent, setOpponent] = useState<ModelId>("transformer");
  const [fen, setFen] = useState(START_FEN);
  const [moves, setMoves] = useState<MoveEvent[]>([]);
  const [status, setStatus] = useState("Ready to play");
  const [running, setRunning] = useState(false);
  const [training, setTraining] = useState(false);
  const [yourTurn, setYourTurn] = useState(false);
  const [legalMoves, setLegalMoves] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [lastUci, setLastUci] = useState<string | null>(null);
  const [view3d, setView3d] = useState(true);
  const [thinkingSide, setThinkingSide] = useState<"white" | "black" | null>(null);
  const [trainReports, setTrainReports] = useState<TrainModelReport[]>([]);
  const [finalResult, setFinalResult] = useState<FinalResult | null>(null);
  const [hoverTip, setHoverTip] = useState<{
    x: number;
    y: number;
    title: string;
    lines: string[];
  } | null>(null);
  const legalMovesRef = useRef<string[]>([]);
  const yourTurnRef = useRef(false);
  const wsRef = useRef<WebSocket | null>(null);
  const onRunningChangeRef = useRef(onRunningChange);
  onRunningChangeRef.current = onRunningChange;

  const wsUrl = useMemo(() => humanSocketUrl("/ws/human"), []);

  const busy = running || training;

  useEffect(() => {
    onRunningChangeRef.current?.(busy);
  }, [busy]);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  const legalTargets = useMemo(() => {
    if (!selected) return [];
    return legalMoves
      .filter((uci) => uci.startsWith(selected))
      .map((uci) => uci.slice(2, 4));
  }, [selected, legalMoves]);

  function ensureSocket(): WebSocket {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return wsRef.current;
    }
    if (wsRef.current && wsRef.current.readyState === WebSocket.CONNECTING) {
      return wsRef.current;
    }
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    ws.onmessage = (msg) => {
      handleEvent(JSON.parse(msg.data) as MatchEvent);
    };
    ws.onerror = () => {
      setStatus("Connection error · is the API running on port 8000?");
      setRunning(false);
      setTraining(false);
    };
    ws.onclose = () => {
      setRunning(false);
      setTraining(false);
      setYourTurn(false);
      yourTurnRef.current = false;
      setThinkingSide(null);
      if (wsRef.current === ws) {
        wsRef.current = null;
      }
    };
    return ws;
  }

  function handleEvent(event: MatchEvent) {
    switch (event.type) {
      case "match_start":
        setFen(event.fen);
        setMoves([]);
        setLastUci(null);
        setSelected(null);
        setLegalMoves([]);
        legalMovesRef.current = [];
        setTrainReports([]);
        setFinalResult(null);
        setTraining(false);
        setYourTurn(false);
        yourTurnRef.current = false;
        setRunning(true);
        setStatus(
          `You are ${event.human_color ?? humanColor} vs ${modelLabel(event.model ?? opponent)}`,
        );
        break;
      case "your_turn":
        setFen(event.fen);
        setYourTurn(true);
        yourTurnRef.current = true;
        setLegalMoves(event.legal_moves ?? []);
        legalMovesRef.current = event.legal_moves ?? [];
        setSelected(null);
        setThinkingSide(null);
        setStatus("Your turn · click a piece, then a highlighted square");
        break;
      case "thinking":
        setYourTurn(false);
        yourTurnRef.current = false;
        setSelected(null);
        setLegalMoves([]);
        legalMovesRef.current = [];
        setThinkingSide(event.side);
        setStatus(`${modelLabel(event.model)} is thinking…`);
        break;
      case "move": {
        setFen(event.fen);
        setMoves((prev) => [...prev, event]);
        setLastUci(event.uci);
        setSelected(null);
        if (event.model === "human") {
          setStatus(`You played ${event.san}`);
        } else {
          setStatus(`${modelLabel(event.model)} played ${event.san}`);
        }
        break;
      }
      case "match_end": {
        setYourTurn(false);
        yourTurnRef.current = false;
        setThinkingSide(null);
        setSelected(null);
        setLegalMoves([]);
        setFen(event.fen);
        setFinalResult({
          result: event.result,
          winner: event.winner,
          termination: event.termination,
          white: event.white || "human",
          black: event.black || opponent,
        });
        setTraining(true);
        setStatus("Game finished · training model…");
        break;
      }
      case "analytics_saved":
        onHistoryChange?.();
        break;
      case "training_start":
        setTraining(true);
        setStatus(`Training ${modelLabel(event.model ?? opponent)}…`);
        break;
      case "training_complete":
        setTrainReports(event.models ?? []);
        setTraining(false);
        setRunning(false);
        setStatus("Game finished · model trained");
        break;
      case "series_end":
        setRunning(false);
        setTraining(false);
        setYourTurn(false);
        yourTurnRef.current = false;
        setThinkingSide(null);
        if (event.result && event.winner && event.termination) {
          setFinalResult({
            result: event.result,
            winner: event.winner,
            termination: event.termination,
            white: event.white || "human",
            black: event.black || opponent,
          });
        }
        setStatus((prev) =>
          prev.includes("model trained") ? prev : "Game finished · model trained",
        );
        break;
      case "match_cancelled":
        setRunning(false);
        setTraining(false);
        setYourTurn(false);
        yourTurnRef.current = false;
        setThinkingSide(null);
        setStatus("Match cancelled");
        break;
      case "error":
        setStatus(event.message);
        if (event.message.toLowerCase().includes("already running")) {
          setRunning(false);
          setTraining(false);
        }
        break;
      default:
        break;
    }
  }

  function startGame() {
    setFen(START_FEN);
    setMoves([]);
    setLastUci(null);
    setSelected(null);
    setLegalMoves([]);
    legalMovesRef.current = [];
    setTrainReports([]);
    setFinalResult(null);
    setTraining(false);
    setYourTurn(false);
    yourTurnRef.current = false;
    setStatus("Connecting…");

    // Fresh socket avoids a stuck previous game blocking moves
    if (wsRef.current) {
      try {
        wsRef.current.send(JSON.stringify({ action: "cancel" }));
      } catch {
        /* ignore */
      }
      wsRef.current.close();
      wsRef.current = null;
    }

    const ws = ensureSocket();
    const payload = {
      action: "start",
      human_color: humanColor,
      model: opponent,
    };
    const sendStart = () => {
      ws.send(JSON.stringify(payload));
      setRunning(true);
    };
    if (ws.readyState === WebSocket.OPEN) {
      sendStart();
    } else {
      ws.onopen = () => sendStart();
    }
  }

  function cancelGame() {
    if (training) return;
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "cancel" }));
    }
  }

  function sendMove(uci: string) {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    setYourTurn(false);
    yourTurnRef.current = false;
    setSelected(null);
    setLegalMoves([]);
    legalMovesRef.current = [];
    ws.send(JSON.stringify({ action: "move", uci }));
  }

  function onSquareClick(sq: string) {
    if (!yourTurnRef.current || training) return;
    const chess = new Chess(fen);
    const piece = chess.get(sq as "a1");

    if (selected) {
      if (selected === sq) {
        setSelected(null);
        return;
      }
      const targets = legalMovesRef.current.filter((uci) => uci.startsWith(selected));
      const match =
        targets.find((uci) => uci.slice(2, 4) === sq && uci.length === 5) ||
        targets.find((uci) => uci.slice(2, 4) === sq);
      if (match) {
        const promos = targets.filter((uci) => uci.slice(2, 4) === sq);
        const queen = promos.find((uci) => uci.endsWith("q"));
        sendMove(queen || match);
        return;
      }
      const turnColor = humanColor === "white" ? "w" : "b";
      if (piece && piece.color === turnColor) {
        setSelected(sq);
        return;
      }
      setSelected(null);
      return;
    }

    const turnColor = humanColor === "white" ? "w" : "b";
    if (piece && piece.color === turnColor) {
      const hasMoves = legalMovesRef.current.some((uci) => uci.startsWith(sq));
      if (hasMoves) setSelected(sq);
    }
  }

  const resultKind =
    finalResult == null
      ? null
      : finalResult.winner === "draw" || finalResult.result === "1/2-1/2"
        ? "draw"
        : finalResult.winner === "human"
          ? "mlp"
          : "transformer";

  return (
    <main className="layout">
      <section className="panel">
        <h2>Human vs Model</h2>
        <label className="field">
          <span className="pairing-label">You play</span>
          <select
            value={humanColor}
            disabled={busy}
            onChange={(e) => setHumanColor(e.target.value as "white" | "black")}
          >
            <option value="white">Ivory Kingdom</option>
            <option value="black">Sun Empire</option>
          </select>
        </label>
        <label className="field">
          <span className="pairing-label">Opponent</span>
          <select
            value={opponent}
            disabled={busy}
            onChange={(e) => setOpponent(e.target.value as ModelId)}
          >
            <option value="mlp">MLP + MCTS</option>
            <option value="transformer">Transformer + MCTS</option>
          </select>
        </label>
        <div className="actions">
          <button className="btn btn-primary" disabled={busy} onClick={startGame}>
            {training ? "Training…" : "Start game"}
          </button>
          <button className="btn btn-ghost" disabled={!running || training} onClick={cancelGame}>
            Cancel
          </button>
        </div>
        <div className="status-pill">
          <span className={`dot ${busy ? "" : "idle"}`} />
          <span className="status-text">{status}</span>
        </div>
        {training && (
          <div className="train-box train-box-live">
            <strong>Training in progress</strong>
            <p>The model is learning from this game. Start stays locked until it finishes.</p>
          </div>
        )}
        {finalResult && (
          <div className={`result-banner result-${resultKind}`}>
            <span className="result-label">Final result</span>
            <strong>{formatResult(finalResult)}</strong>
            <span className="result-score">{finalResult.result}</span>
          </div>
        )}
        {!training && trainReports.length > 0 && (
          <div className="train-box">
            <strong>Training complete</strong>
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
          <div className={`player ${thinkingSide === "white" || (yourTurn && humanColor === "white") ? "active" : ""}`}>
            <span className="side-white">Ivory Kingdom</span>
            <strong>{humanColor === "white" ? "You" : modelLabel(opponent)}</strong>
          </div>
          <button
            type="button"
            className="btn btn-ghost view-toggle"
            onClick={() => setView3d((v) => !v)}
          >
            {view3d ? "2D board" : "3D board"}
          </button>
          <div className={`player ${thinkingSide === "black" || (yourTurn && humanColor === "black") ? "active" : ""}`}>
            <span className="side-black">Sun Empire</span>
            <strong>{humanColor === "black" ? "You" : modelLabel(opponent)}</strong>
          </div>
        </div>
        <ChessBoard
          fen={fen}
          lastUci={lastUci}
          selected={selected}
          legalTargets={legalTargets}
          interactive={yourTurn && !training}
          orientation={humanColor}
          view3d={view3d}
          onSquareClick={onSquareClick}
        />
      </section>

      <section className="panel">
        <h2>Moves</h2>
        <ol className="move-list">
          {moves.map((m) => (
            <li
              key={m.ply}
              className="move-item"
              onMouseEnter={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const tipWidth = 280;
                const x = Math.min(Math.max(8, rect.left), window.innerWidth - tipWidth - 8);
                const y = rect.bottom + 8;
                setHoverTip({
                  x,
                  y,
                  title: `Ply ${m.ply} · ${m.san}`,
                  lines: [
                    `${m.side === "white" ? "Ivory Kingdom" : "Sun Empire"} · ${
                      m.model === "human" ? "You" : modelLabel(m.model)
                    }`,
                    `Played ${m.san} (${m.uci})`,
                    explainSan(m.san, m.uci),
                  ],
                });
              }}
              onMouseLeave={() => setHoverTip(null)}
            >
              <span>{m.ply}.</span>
              <span className="san">{m.san}</span>
              <span className="model">{m.model === "human" ? "You" : modelLabel(m.model)}</span>
            </li>
          ))}
        </ol>
        {hoverTip &&
          createPortal(
            <div
              className="move-tip-float"
              style={{ left: hoverTip.x, top: hoverTip.y }}
              role="tooltip"
            >
              <strong>{hoverTip.title}</strong>
              {hoverTip.lines.map((line) => (
                <div key={line}>{line}</div>
              ))}
            </div>,
            document.body,
          )}
      </section>
    </main>
  );
}
