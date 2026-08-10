import { useEffect, useMemo, useRef, useState } from "react";
import { ChessBoard } from "./ChessBoard";
import { modelLabel, type MatchEvent, type ModelId, type MoveEvent } from "./types";

const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

export default function App() {
  const [whiteModel, setWhiteModel] = useState<ModelId>("mlp");
  const [blackModel, setBlackModel] = useState<ModelId>("transformer");
  const [simulations, setSimulations] = useState(64);
  const [fen, setFen] = useState(START_FEN);
  const [moves, setMoves] = useState<MoveEvent[]>([]);
  const [status, setStatus] = useState("Ready for a duel");
  const [running, setRunning] = useState(false);
  const [thinkingSide, setThinkingSide] = useState<"white" | "black" | null>(null);
  const [lastUci, setLastUci] = useState<string | null>(null);
  const [latestMcts, setLatestMcts] = useState<MoveEvent["mcts"] | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const wsUrl = useMemo(() => {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const host = window.location.host;
    return `${proto}://${host}/ws/match`;
  }, []);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  function ensureSocket(): WebSocket {
    if (wsRef.current && wsRef.current.readyState <= WebSocket.OPEN) {
      return wsRef.current;
    }
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (msg) => {
      const event = JSON.parse(msg.data) as MatchEvent;
      handleEvent(event);
    };
    ws.onclose = () => {
      setRunning(false);
      setThinkingSide(null);
    };
    return ws;
  }

  function handleEvent(event: MatchEvent) {
    switch (event.type) {
      case "match_start":
        setFen(event.fen);
        setMoves([]);
        setLastUci(null);
        setLatestMcts(null);
        setRunning(true);
        setStatus(`${modelLabel(event.white)} vs ${modelLabel(event.black)}`);
        break;
      case "thinking":
        setThinkingSide(event.side);
        setStatus(`${modelLabel(event.model)} is searching…`);
        break;
      case "move":
        setFen(event.fen);
        setMoves((prev) => [...prev, event]);
        setLastUci(event.uci);
        setLatestMcts(event.mcts);
        setThinkingSide(null);
        setStatus(`${modelLabel(event.model)} played ${event.san}`);
        break;
      case "match_end":
        setRunning(false);
        setThinkingSide(null);
        setFen(event.fen);
        setStatus(`Result ${event.result} · ${event.winner}`);
        break;
      case "match_cancelled":
        setRunning(false);
        setThinkingSide(null);
        setStatus("Match cancelled");
        break;
      case "error":
        setRunning(false);
        setStatus(event.message);
        break;
      default:
        break;
    }
  }

  function startMatch() {
    setFen(START_FEN);
    setMoves([]);
    setLastUci(null);
    setLatestMcts(null);
    setStatus("Connecting…");
    const ws = ensureSocket();

    const payload = {
      action: "start",
      white_model: whiteModel,
      black_model: blackModel,
      simulations,
    };

    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
      setRunning(true);
    } else {
      ws.onopen = () => {
        ws.send(JSON.stringify(payload));
        setRunning(true);
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
    <div className="app">
      <header className="hero">
        <h1 className="brand">Ply Arena</h1>
        <p className="tagline">
          Watch MLP+MCTS and Transformer+MCTS duel under a shared AlphaZero-style search.
        </p>
      </header>

      <main className="layout">
        <section className="panel">
          <h2>Match setup</h2>
          <div className="field">
            <label htmlFor="white">White</label>
            <select
              id="white"
              value={whiteModel}
              disabled={running}
              onChange={(e) => setWhiteModel(e.target.value as ModelId)}
            >
              <option value="mlp">MLP + MCTS</option>
              <option value="transformer">Transformer + MCTS</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="black">Black</label>
            <select
              id="black"
              value={blackModel}
              disabled={running}
              onChange={(e) => setBlackModel(e.target.value as ModelId)}
            >
              <option value="transformer">Transformer + MCTS</option>
              <option value="mlp">MLP + MCTS</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="sims">MCTS simulations / move</label>
            <input
              id="sims"
              type="number"
              min={8}
              max={400}
              step={8}
              disabled={running}
              value={simulations}
              onChange={(e) => setSimulations(Number(e.target.value))}
            />
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
          <h2>Moves</h2>
          <ol className="move-list">
            {moves.map((m) => (
              <li key={m.ply}>
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
    </div>
  );
}
