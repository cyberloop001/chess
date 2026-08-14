import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
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
  if (final.winner === "mlp") {
    return `MLP + MCTS wins · ${final.result} · ${why}`;
  }
  if (final.winner === "transformer") {
    return `Transformer + MCTS wins · ${final.result} · ${why}`;
  }
  return `${final.result} · ${why}`;
}

function explainMove(m: MoveEvent): { title: string; lines: string[] } {
  const side = m.side === "white" ? "White" : "Black";
  const lines = [
    `${side} · ${modelLabel(m.model)}`,
    `Played ${m.san} (${m.uci})`,
    explainSan(m.san, m.uci),
  ];
  const mcts = m.mcts;
  if (mcts) {
    lines.push(`MCTS root value ${mcts.root_value} · ${mcts.simulations} sims`);
    if (mcts.top_moves?.length) {
      lines.push("Top candidates:");
      for (const tm of mcts.top_moves.slice(0, 3)) {
        lines.push(`${tm.uci} · visits ${tm.visits} · Q ${tm.q}`);
      }
    }
  } else {
    lines.push("No MCTS details for this move.");
  }
  return { title: `Ply ${m.ply} · ${m.san}`, lines };
}

export function ArenaPage({ onHistoryChange, onRunningChange }: Props) {
  const [whiteModel, setWhiteModel] = useState<ModelId>("mlp");
  const [blackModel, setBlackModel] = useState<ModelId>("transformer");
  const [trainCount, setTrainCount] = useState(1);
  const [seriesGame, setSeriesGame] = useState(0);
  const [seriesTotal, setSeriesTotal] = useState(1);
  const [fen, setFen] = useState(START_FEN);
  const [moves, setMoves] = useState<MoveEvent[]>([]);
  const [status, setStatus] = useState("Ready for a duel");
  const [running, setRunning] = useState(false);
  const [training, setTraining] = useState(false);
  const [thinkingSide, setThinkingSide] = useState<"white" | "black" | null>(null);
  const [lastUci, setLastUci] = useState<string | null>(null);
  const [latestMcts, setLatestMcts] = useState<MoveEvent["mcts"] | null>(null);
  const [trainReports, setTrainReports] = useState<TrainModelReport[]>([]);
  const [finalResult, setFinalResult] = useState<FinalResult | null>(null);
  const [hoverTip, setHoverTip] = useState<{
    x: number;
    y: number;
    title: string;
    lines: string[];
  } | null>(null);
  const matchMetaRef = useRef({ white: "mlp", black: "transformer", simulations: 64 });
  const movesRef = useRef<MoveEvent[]>([]);
  const pendingResultRef = useRef<FinalResult | null>(null);
  const seriesTotalRef = useRef(1);
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
  }

  const busy = running || training;

  useEffect(() => {
    onRunningChangeRef.current?.(busy);
  }, [busy]);

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
      setTraining(false);
      setThinkingSide(null);
    };
    return ws;
  }

  function handleEvent(event: MatchEvent) {
    switch (event.type) {
      case "series_start": {
        const total = event.train_count ?? event.max_games ?? 1;
        seriesTotalRef.current = total;
        setSeriesTotal(total);
        setSeriesGame(0);
        setRunningState(true);
        setTraining(false);
        setTrainReports([]);
        setFinalResult(null);
        pendingResultRef.current = null;
        setStatus(
          total > 1
            ? `Training series · ${total} games · ${modelLabel(event.white)} vs ${modelLabel(event.black)}`
            : `${modelLabel(event.white)} vs ${modelLabel(event.black)}`,
        );
        break;
      }
      case "match_start": {
        const idx = event.game_index ?? 1;
        const total = seriesTotalRef.current;
        setSeriesGame(idx);
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
        setTraining(false);
        setRunningState(true);
        setStatus(
          total > 1
            ? `Game ${idx} of ${total} · ${modelLabel(event.white)} (White) vs ${modelLabel(event.black)} (Black)`
            : `${modelLabel(event.white)} (White) vs ${modelLabel(event.black)} (Black)`,
        );
        break;
      }
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
        setTraining(true);
        {
          const idx = event.game_index ?? (seriesGame || 1);
          const total = seriesTotalRef.current;
          setStatus(
            total > 1
              ? `Game ${idx} of ${total} finished · training models…`
              : "Game finished · training models…",
          );
        }
        break;
      }
      case "analytics_saved":
        onHistoryChange?.();
        break;
      case "training_start": {
        const total = event.train_count ?? seriesTotalRef.current;
        setTraining(true);
        setStatus(
          total > 1
            ? `Training after game ${event.game ?? seriesGame} of ${total}…`
            : "Training models… please wait",
        );
        break;
      }
      case "training_complete": {
        const total = event.train_count ?? seriesTotalRef.current;
        const gameNum = event.game ?? 0;
        setTrainReports(event.models);
        setTraining(false);
        setStatus(
          total > 1
            ? `Trained ${gameNum} of ${total}` +
              (gameNum < total ? " · starting next duel…" : " · series complete")
            : "Game finished · models trained",
        );
        break;
      }
      case "series_end": {
        setRunningState(false);
        setTraining(false);
        setThinkingSide(null);
        const end: FinalResult = pendingResultRef.current ?? {
          result: event.result || "unknown",
          winner: event.series_winner,
          termination: event.termination || "unknown",
          white: event.white || matchMetaRef.current.white,
          black: event.black || matchMetaRef.current.black,
        };
        setFinalResult(end);
        const played = event.game_count;
        const total = event.train_count ?? seriesTotalRef.current;
        setStatus(
          total > 1
            ? `Series finished · ${played} of ${total} games trained`
            : "Game finished · models trained",
        );
        break;
      }
      case "match_cancelled":
        setRunningState(false);
        setTraining(false);
        setThinkingSide(null);
        setStatus("Match cancelled");
        break;
      case "error":
        setRunningState(false);
        setTraining(false);
        setStatus(event.message);
        break;
      default:
        break;
    }
  }

  function startMatch() {
    const n = Math.max(1, Math.min(100, Math.floor(Number(trainCount) || 1)));
    setTrainCount(n);
    seriesTotalRef.current = n;
    setSeriesTotal(n);
    setSeriesGame(0);
    setFen(START_FEN);
    setMoves([]);
    movesRef.current = [];
    setLastUci(null);
    setLatestMcts(null);
    setTrainReports([]);
    setFinalResult(null);
    pendingResultRef.current = null;
    setTraining(false);
    setStatus(n > 1 ? `Connecting · ${n} train cycles…` : "Connecting…");
    const ws = ensureSocket();
    const payload = {
      action: "start",
      white_model: "mlp",
      black_model: "transformer",
      train_count: n,
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
    if (training) return;
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
        <label className="field">
          <span className="pairing-label">Train count</span>
          <input
            type="number"
            min={1}
            max={100}
            step={1}
            value={trainCount}
            disabled={busy}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (!Number.isFinite(v)) {
                setTrainCount(1);
                return;
              }
              setTrainCount(Math.max(1, Math.min(100, Math.floor(v))));
            }}
          />
          <span className="field-hint">Games to play and train (1–100)</span>
        </label>
        <div className="actions">
          <button className="btn btn-primary" disabled={busy} onClick={startMatch}>
            {training ? "Training…" : seriesTotal > 1 && running ? "Series running…" : "Start duel"}
          </button>
          <button
            className="btn btn-ghost"
            disabled={!running || training}
            onClick={cancelMatch}
          >
            Cancel
          </button>
        </div>
        <div className="status-pill">
          <span className={`dot ${busy ? "" : "idle"}`} />
          <span className="status-text">{status}</span>
        </div>
        {(running || training || seriesGame > 0) && seriesTotal > 1 && (
          <div className="series-progress">
            <strong>
              Progress · {Math.min(seriesGame, seriesTotal)} / {seriesTotal}
            </strong>
            <div className="series-bar" aria-hidden>
              <div
                className="series-bar-fill"
                style={{ width: `${(Math.min(seriesGame, seriesTotal) / seriesTotal) * 100}%` }}
              />
            </div>
          </div>
        )}
        {training && (
          <div className="train-box train-box-live">
            <strong>Training in progress</strong>
            <p>
              Both models are updating from this game
              {seriesTotal > 1 ? ` (${seriesGame} of ${seriesTotal})` : ""}. Start stays locked
              until the full series finishes.
            </p>
          </div>
        )}
        {finalResult && (
          <div className={`result-banner result-${resultKind}`}>
            <span className="result-label">
              {seriesTotal > 1 && seriesGame > 0 ? `Game ${seriesGame} result` : "Final result"}
            </span>
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
      </section>

      <section className="panel">
        <h2>Moves{seriesTotal > 1 && seriesGame > 0 ? ` · game ${seriesGame}` : ""}</h2>
        <ol className="move-list">
          {moves.map((m) => (
            <li
              key={`${m.game_index ?? 0}-${m.ply}`}
              className="move-item"
              onMouseEnter={(e) => {
                const tip = explainMove(m);
                const rect = e.currentTarget.getBoundingClientRect();
                const tipWidth = 280;
                const x = Math.min(Math.max(8, rect.left), window.innerWidth - tipWidth - 8);
                const y = rect.bottom + 8;
                setHoverTip({ x, y, title: tip.title, lines: tip.lines });
              }}
              onMouseLeave={() => setHoverTip(null)}
            >
              <span>{m.ply}.</span>
              <span className="san">{m.san}</span>
              <span className="model">{modelLabel(m.model)}</span>
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
