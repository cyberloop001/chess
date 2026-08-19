import { useEffect, useMemo, useRef, useState } from "react";
import {
  SELFPLAY_CITATIONS,
  clearSelfPlayHistoryRemote,
  fetchSelfPlayHistory,
  improvementFor,
} from "./selfPlayHistory";
import { shortModel, type ModelId, type SelfPlayEvent, type SelfPlayRecord } from "./types";

function formatWhen(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function pct(n: number): string {
  const sign = n > 0 ? "+" : "";
  return `${sign}${Math.round(n * 100)}%`;
}

function LossChart({
  title,
  series,
}: {
  title: string;
  series: Array<{ label: string; color: string; points: number[] }>;
}) {
  const width = 640;
  const height = 220;
  const pad = { t: 16, r: 16, b: 28, l: 48 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;
  const all = series.flatMap((s) => s.points);
  const minY = 0;
  const maxY = all.length ? Math.max(0.05, ...all) * 1.08 : 1;
  const span = maxY - minY || 1;
  const ticks = [0, maxY * 0.25, maxY * 0.5, maxY * 0.75, maxY];

  function x(i: number, len: number) {
    if (len <= 1) return pad.l;
    return pad.l + (i / (len - 1)) * innerW;
  }
  function y(v: number) {
    return pad.t + ((maxY - v) / span) * innerH;
  }

  return (
    <div className="chart-block">
      <h3>{title}</h3>
      <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        {ticks.map((tick) => (
          <text key={tick} x={pad.l - 8} y={y(tick) + 4} className="chart-axis" textAnchor="end">
            {tick.toFixed(2)}
          </text>
        ))}
        {series.map((s) => {
          if (s.points.length === 0) return null;
          const d = s.points
            .map((v, i) => `${i === 0 ? "M" : "L"} ${x(i, s.points.length)} ${y(v)}`)
            .join(" ");
          return <path key={s.label} d={d} fill="none" stroke={s.color} strokeWidth="2.5" />;
        })}
        <text x={pad.l} y={height - 8} className="chart-axis">
          Self-play game →
        </text>
      </svg>
      <div className="chart-legend">
        {series.map((s) => (
          <span key={s.label}>
            <i style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}

function Delta({ label, value, invert }: { label: string; value: number; invert?: boolean }) {
  const better = invert ? value < 0 : value > 0;
  const cls = value === 0 ? "" : better ? "delta-up" : "delta-down";
  return (
    <div className={`delta ${cls}`}>
      <span>{label}</span>
      <strong>{pct(value)}</strong>
    </div>
  );
}

type Props = {
  live?: boolean;
};

export function SelfPlayPanel({ live: liveProp }: Props) {
  const [games, setGames] = useState<SelfPlayRecord[]>([]);
  const [model, setModel] = useState<ModelId>("transformer");
  const [gameCount, setGameCount] = useState(4);
  const [status, setStatus] = useState("Idle");
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState({ game: 0, total: 0 });
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const wsUrl = useMemo(() => {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${window.location.host}/ws/selfplay`;
  }, []);

  const refresh = async () => {
    const next = await fetchSelfPlayHistory();
    setGames(next);
  };

  useEffect(() => {
    void refresh().catch((err: unknown) => {
      setError(err instanceof Error ? err.message : "Failed to load self-play history");
    });
  }, []);

  useEffect(() => {
    return () => wsRef.current?.close();
  }, []);

  const mlpImp = useMemo(() => improvementFor(games, "mlp"), [games]);
  const trImp = useMemo(() => improvementFor(games, "transformer"), [games]);

  const lossSeries = useMemo(() => {
    const chrono = [...games].sort((a, b) => a.playedAt.localeCompare(b.playedAt));
    return [
      {
        label: "MLP total loss",
        color: "#ff9a3c",
        points: chrono.filter((g) => g.model === "mlp").map((g) => g.totalLoss),
      },
      {
        label: "Transformer total loss",
        color: "#7ec8ff",
        points: chrono.filter((g) => g.model === "transformer").map((g) => g.totalLoss),
      },
    ];
  }, [games]);

  const policySeries = useMemo(() => {
    const chrono = [...games].sort((a, b) => a.playedAt.localeCompare(b.playedAt));
    return [
      {
        label: "MLP policy CE",
        color: "#ff9a3c",
        points: chrono.filter((g) => g.model === "mlp").map((g) => g.policyLoss),
      },
      {
        label: "Transformer policy CE",
        color: "#7ec8ff",
        points: chrono.filter((g) => g.model === "transformer").map((g) => g.policyLoss),
      },
    ];
  }, [games]);

  function handleEvent(event: SelfPlayEvent) {
    if (event.type === "selfplay_start") {
      setRunning(true);
      setProgress({ game: 0, total: event.games });
      setStatus(`Self-play ${shortModel(event.model)} · 0 / ${event.games}`);
    } else if (event.type === "selfplay_thinking") {
      setProgress({ game: event.game, total: event.games });
      setStatus(`Playing game ${event.game} / ${event.games} · ${shortModel(event.model)}`);
    } else if (event.type === "selfplay_game") {
      const rec = event as SelfPlayRecord;
      setGames((prev) => [rec, ...prev.filter((g) => g.id !== rec.id)]);
      setProgress({ game: event.game, total: event.games });
      setStatus(
        `Trained game ${event.game} / ${event.games} · loss ${event.totalLoss.toFixed(3)} · ${event.termination}`,
      );
    } else if (event.type === "selfplay_complete" || event.type === "selfplay_cancelled") {
      setRunning(false);
      setStatus(event.type === "selfplay_cancelled" ? "Run cancelled" : "Self-play run complete");
      void refresh().catch(() => undefined);
    } else if (event.type === "error") {
      setError(event.message);
      setRunning(false);
    } else if (event.type === "info") {
      setStatus(event.message);
    }
  }

  function startRun() {
    setError(null);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    ws.onopen = () => {
      ws.send(JSON.stringify({ action: "start", model, games: gameCount }));
    };
    ws.onmessage = (msg) => {
      try {
        handleEvent(JSON.parse(msg.data) as SelfPlayEvent);
      } catch {
        setError("Bad self-play event");
      }
    };
    ws.onerror = () => {
      setError("Self-play socket error · is the API running on port 8000?");
      setRunning(false);
    };
  }

  function cancelRun() {
    wsRef.current?.send(JSON.stringify({ action: "cancel" }));
  }

  const live = liveProp ?? running;
  const pctDone = progress.total > 0 ? Math.round((progress.game / progress.total) * 100) : 0;

  return (
    <section className="selfplay-dash">
      <div className="analytics-hero-row">
        <div>
          <h2 className="analytics-title">Self-play training</h2>
          <p className="analytics-sub">
            Each net trains on its own self-play games. Lower loss is better. Improvement compares
            the first vs last quarter of logged games.
          </p>
        </div>
        <form
          className="selfplay-controls"
          onSubmit={(e) => {
            e.preventDefault();
            if (!running) startRun();
          }}
        >
          <label>
            Model
            <select value={model} disabled={running} onChange={(e) => setModel(e.target.value as ModelId)}>
              <option value="transformer">Transformer</option>
              <option value="mlp">MLP</option>
            </select>
          </label>
          <label>
            Games
            <input
              type="number"
              min={1}
              max={100}
              value={gameCount}
              disabled={running}
              onChange={(e) => setGameCount(Math.max(1, Number(e.target.value) || 1))}
            />
          </label>
          {running ? (
            <button type="button" className="btn btn-ghost" onClick={cancelRun}>
              Cancel
            </button>
          ) : (
            <button type="submit" className="btn btn-primary">
              Start self-play
            </button>
          )}
          <button
            type="button"
            className="btn btn-ghost"
            disabled={busy || games.length === 0}
            onClick={() => {
              setBusy(true);
              void clearSelfPlayHistoryRemote()
                .then(() => setGames([]))
                .finally(() => setBusy(false));
            }}
          >
            Clear log
          </button>
        </form>
      </div>

      <p className="muted-note">
        {status}
        {live && progress.total > 0 ? ` · ${pctDone}%` : ""}
      </p>
      {running && (
        <div className="bar-track" aria-label="Self-play progress">
          <div className="bar-left" style={{ width: `${pctDone}%` }} />
        </div>
      )}
      {error && <p className="muted-note">{error}</p>}

      <section className="stat-grid">
        <article className="stat">
          <span className="stat-label">Logged games</span>
          <strong>{games.length}</strong>
        </article>
        <article className="stat">
          <span className="stat-label">Transformer loss drop</span>
          <strong>{trImp ? pct(trImp.lossDropPct) : "—"}</strong>
        </article>
        <article className="stat">
          <span className="stat-label">MLP loss drop</span>
          <strong>{mlpImp ? pct(mlpImp.lossDropPct) : "—"}</strong>
        </article>
        <article className="stat">
          <span className="stat-label">Transformer decisive Δ</span>
          <strong>{trImp ? pct(trImp.decisiveDelta) : "—"}</strong>
        </article>
      </section>

      <section className="analytics-grid">
        <div className="panel">
          <h3>How much improved</h3>
          <p className="muted-note">
            Early = first 25% of that model’s self-play log. Late = last 25%. Policy/value losses
            follow AlphaZero (citations below).
          </p>
          {[trImp, mlpImp].map((imp) =>
            imp ? (
              <div key={imp.id} className="improve-card">
                <strong>{shortModel(imp.id)}</strong>
                <span className="muted-note">{imp.games} games in log</span>
                <Delta label="Total loss drop" value={imp.lossDropPct} />
                <Delta label="Decisive-game rate" value={imp.decisiveDelta} />
                <Delta label="Avg plies down" value={imp.plyDelta / Math.max(1, imp.earlyPlies)} />
                <p className="muted-note">
                  Loss {imp.earlyLoss.toFixed(3)} → {imp.lateLoss.toFixed(3)} · plies{" "}
                  {imp.earlyPlies.toFixed(0)} → {imp.latePlies.toFixed(0)}
                </p>
              </div>
            ) : null,
          )}
          {!trImp && !mlpImp && (
            <p className="muted-note">Need at least two self-play games per model to measure change.</p>
          )}
        </div>
        <div className="panel">
          <h3>Citations</h3>
          <ol className="cite-list">
            {SELFPLAY_CITATIONS.map((c, i) => (
              <li key={c.id}>
                <a href={c.href} target="_blank" rel="noreferrer">
                  [{i + 1}] {c.title}
                </a>
                <div className="muted-note">{c.note}</div>
              </li>
            ))}
            <li>
              [3] This dashboard · {games.length} stored games in <code>data/selfplay_history.json</code>
              {games[0] ? ` · latest ${formatWhen(games[0].playedAt)}` : ""}.
            </li>
          </ol>
        </div>
      </section>

      {games.length > 0 && (
        <section className="analytics-grid analytics-grid-wide">
          <div className="panel">
            <LossChart title="Total loss (policy CE + value MSE)" series={lossSeries} />
            <LossChart title="Policy loss vs MCTS π" series={policySeries} />
            <p className="muted-note">
              Cited in [1]: loss is cross-entropy to search policy plus MSE to outcome z. A drop
              means the net matches its own MCTS more closely — not Elo vs the other architecture.
            </p>
          </div>
          <div className="panel">
            <h3>Recent self-play games</h3>
            <div className="match-table-wrap">
              <table className="match-table">
                <thead>
                  <tr>
                    <th>When</th>
                    <th>Model</th>
                    <th>Result</th>
                    <th>End</th>
                    <th>Plies</th>
                    <th>Loss</th>
                  </tr>
                </thead>
                <tbody>
                  {games.slice(0, 24).map((g) => (
                    <tr key={g.id}>
                      <td>{formatWhen(g.playedAt)}</td>
                      <td>{shortModel(g.model)}</td>
                      <td>{g.result}</td>
                      <td>{g.termination}</td>
                      <td>{g.plies}</td>
                      <td>{g.totalLoss.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      )}
    </section>
  );
}
