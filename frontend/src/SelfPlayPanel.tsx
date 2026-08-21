import { useEffect, useMemo, useRef, useState } from "react";
import { wsUrl as selfPlaySocketUrl } from "./api";
import { ChessBoard } from "./ChessBoard";
import {
  SELFPLAY_CITATIONS,
  clearSelfPlayHistoryRemote,
  fetchSelfPlayHistory,
  improvementFor,
} from "./selfPlayHistory";
import { shortModel, type ModelId, type SelfPlayEvent, type SelfPlayRecord } from "./types";

const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

type LiveMove = {
  ply: number;
  side: "white" | "black";
  uci: string;
  san: string;
  root_value: number;
  net_value: number;
  move_q: number;
};

function formatEval(v: number): string {
  // Fixed width so Root / Net / Move Q columns stay aligned with headers.
  return (Object.is(v, -0) ? 0 : v).toFixed(4);
}

type TrainStepPoint = {
  step: number;
  epoch: number;
  policy_loss: number;
  value_loss: number;
  total_loss: number;
  grad_norm: number;
};

type LayerDelta = {
  name: string;
  short: string;
  numel: number;
  mean_abs_delta: number;
  rms_delta: number;
  rel_mean: number;
};

type WeightHist = {
  counts: number[];
  edges: number[];
  mean_rel: number;
  median_rel: number;
  sampled: number;
};

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
  xLabel = "Self-play game →",
}: {
  title: string;
  series: Array<{ label: string; color: string; points: number[] }>;
  xLabel?: string;
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
          {xLabel}
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

function LayerDeltaChart({
  layers,
}: {
  layers: Array<{ short: string; rel_mean: number; numel: number }>;
}) {
  const top = layers.slice(0, 18);
  const width = 640;
  const rowH = 18;
  const pad = { t: 8, r: 16, b: 8, l: 150 };
  const height = pad.t + pad.b + Math.max(1, top.length) * rowH;
  const innerW = width - pad.l - pad.r;
  const maxV = Math.max(1e-8, ...top.map((l) => l.rel_mean));

  return (
    <div className="chart-block">
      <h3>Weight update by layer · mean |Δw| / (|w|+ε)</h3>
      <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Layer weight deltas">
        {top.map((layer, i) => {
          const y = pad.t + i * rowH;
          const w = (layer.rel_mean / maxV) * innerW;
          return (
            <g key={`${layer.short}-${i}`}>
              <text x={pad.l - 8} y={y + 12} className="chart-axis" textAnchor="end">
                {layer.short}
              </text>
              <rect x={pad.l} y={y + 3} width={Math.max(2, w)} height={12} fill="#7ec8ff" opacity={0.85} />
              <text x={pad.l + Math.max(2, w) + 6} y={y + 12} className="chart-axis">
                {layer.rel_mean.toExponential(2)}
              </text>
            </g>
          );
        })}
      </svg>
      <p className="muted-note">
        Every trainable tensor is updated by Adam each step. Bars show relative change after this
        game’s training pass (top {top.length} of {layers.length} tensors).
      </p>
    </div>
  );
}

function WeightHistChart({
  hist,
}: {
  hist: { counts: number[]; edges: number[]; mean_rel: number; median_rel: number; sampled: number };
}) {
  const width = 640;
  const height = 200;
  const pad = { t: 16, r: 16, b: 36, l: 48 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;
  const maxC = Math.max(1, ...hist.counts);
  const n = hist.counts.length;

  return (
    <div className="chart-block">
      <h3>Distribution of relative weight updates</h3>
      <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Weight update histogram">
        {hist.counts.map((c, i) => {
          const bw = innerW / Math.max(1, n);
          const h = (c / maxC) * innerH;
          const x = pad.l + i * bw;
          const y = pad.t + innerH - h;
          return <rect key={i} x={x + 1} y={y} width={Math.max(1, bw - 2)} height={h} fill="#ff9a3c" opacity={0.8} />;
        })}
        <text x={pad.l} y={height - 10} className="chart-axis">
          |Δw| / |w| (log bins) →
        </text>
        <text x={width - pad.r} y={height - 10} className="chart-axis" textAnchor="end">
          larger updates
        </text>
      </svg>
      <p className="muted-note">
        Across {hist.sampled.toLocaleString()} weight entries · mean rel{" "}
        {hist.mean_rel.toExponential(2)} · median {hist.median_rel.toExponential(2)}. Most mass near
        zero means tiny Adam steps; a right tail means some tensors moved more.
      </p>
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
  const [liveFen, setLiveFen] = useState(START_FEN);
  const [lastUci, setLastUci] = useState<string | null>(null);
  const [liveMoves, setLiveMoves] = useState<LiveMove[]>([]);
  const [liveMeta, setLiveMeta] = useState<{
    model: string;
    game: number;
    games: number;
    rootValue: number | null;
    phase: "idle" | "playing" | "training";
  }>({ model: "transformer", game: 0, games: 0, rootValue: null, phase: "idle" });
  const [trainTrace, setTrainTrace] = useState<TrainStepPoint[]>([]);
  const [layerDeltas, setLayerDeltas] = useState<LayerDelta[]>([]);
  const [weightHist, setWeightHist] = useState<WeightHist | null>(null);
  const [weightMeta, setWeightMeta] = useState<{
    game: number;
    steps: number;
    samples: number;
    lr: number;
    epochs: number;
  } | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const moveListRef = useRef<HTMLDivElement | null>(null);

  const wsUrl = useMemo(() => selfPlaySocketUrl("/ws/selfplay"), []);

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

  useEffect(() => {
    moveListRef.current?.scrollTo({ top: moveListRef.current.scrollHeight });
  }, [liveMoves.length]);

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
      setLiveFen(START_FEN);
      setLastUci(null);
      setLiveMoves([]);
      setTrainTrace([]);
      setLayerDeltas([]);
      setWeightHist(null);
      setWeightMeta(null);
      setLiveMeta({
        model: event.model,
        game: 0,
        games: event.games,
        rootValue: null,
        phase: "playing",
      });
      setStatus(`Self-play ${shortModel(event.model)} · 0 / ${event.games}`);
    } else if (event.type === "selfplay_thinking") {
      setProgress({ game: event.game, total: event.games });
      setLiveFen(event.fen ?? START_FEN);
      setLastUci(null);
      setLiveMoves([]);
      setTrainTrace([]);
      setLiveMeta({
        model: event.model,
        game: event.game,
        games: event.games,
        rootValue: null,
        phase: "playing",
      });
      setStatus(`Playing game ${event.game} / ${event.games} · ${shortModel(event.model)}`);
    } else if (event.type === "selfplay_move") {
      setLiveFen(event.fen);
      setLastUci(event.uci);
      setLiveMoves((prev) => [
        ...prev,
        {
          ply: event.ply,
          side: event.side,
          uci: event.uci,
          san: event.san,
          root_value: event.root_value,
          net_value: event.net_value ?? event.root_value,
          move_q: event.move_q ?? event.root_value,
        },
      ]);
      setLiveMeta({
        model: event.model,
        game: event.game,
        games: event.games,
        rootValue: event.root_value,
        phase: "playing",
      });
      setProgress({ game: event.game, total: event.games });
      setStatus(
        `Game ${event.game}/${event.games} · ply ${event.ply} · ${event.san} · root ${formatEval(event.root_value)} · net ${formatEval(event.net_value ?? event.root_value)}`,
      );
    } else if (event.type === "selfplay_training") {
      if (event.fen) setLiveFen(event.fen);
      setTrainTrace([]);
      setLiveMeta((m) => ({ ...m, phase: "training", game: event.game, games: event.games }));
      setStatus(
        `Training on game ${event.game}/${event.games} · ${event.plies} plies · ${event.termination}`,
      );
    } else if (event.type === "selfplay_train_step") {
      setLiveMeta((m) => ({ ...m, phase: "training", game: event.game, games: event.games }));
      setTrainTrace((prev) => [
        ...prev,
        {
          step: event.step,
          epoch: event.epoch,
          policy_loss: event.policy_loss,
          value_loss: event.value_loss,
          total_loss: event.total_loss,
          grad_norm: event.grad_norm,
        },
      ]);
      setStatus(
        `Train step ${event.step} · epoch ${event.epoch}/${event.epochs} · loss ${event.total_loss.toFixed(3)} · ‖∇‖ ${event.grad_norm.toFixed(3)}`,
      );
    } else if (event.type === "selfplay_weights") {
      setLayerDeltas(event.layer_deltas);
      setWeightHist(event.weight_hist);
      setWeightMeta({
        game: event.game,
        steps: event.steps,
        samples: event.samples,
        lr: event.lr,
        epochs: event.epochs,
      });
      if (event.train_trace?.length) {
        setTrainTrace(
          event.train_trace.map((p) => ({
            step: p.step,
            epoch: p.epoch,
            policy_loss: p.policy_loss,
            value_loss: p.value_loss,
            total_loss: p.total_loss,
            grad_norm: p.grad_norm,
          })),
        );
      }
      setStatus(
        `Updated ${event.layer_deltas.length} weight tensors · ${event.steps} Adam steps · game ${event.game}`,
      );
    } else if (event.type === "selfplay_game") {
      const rec = event as SelfPlayRecord;
      setGames((prev) => [rec, ...prev.filter((g) => g.id !== rec.id)]);
      setProgress({ game: event.game, total: event.games });
      setStatus(
        `Trained game ${event.game} / ${event.games} · loss ${event.totalLoss.toFixed(3)} · ${event.termination}`,
      );
    } else if (event.type === "selfplay_complete" || event.type === "selfplay_cancelled") {
      setRunning(false);
      setLiveMeta((m) => ({ ...m, phase: "idle" }));
      setStatus(event.type === "selfplay_cancelled" ? "Run cancelled" : "Self-play run complete");
      void refresh().catch(() => undefined);
    } else if (event.type === "error") {
      setError(event.message);
      setRunning(false);
      setLiveMeta((m) => ({ ...m, phase: "idle" }));
    } else if (event.type === "info") {
      setStatus(event.message);
    }
  }

  function startRun() {
    setError(null);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          action: "start",
          model,
          games: gameCount,
          simulations: 256,
          move_delay_ms: 120,
        }),
      );
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

  const trainLossSeries = useMemo(
    () => [
      {
        label: "Total loss",
        color: "#e8dcc8",
        points: trainTrace.map((p) => p.total_loss),
      },
      {
        label: "Policy CE",
        color: "#7ec8ff",
        points: trainTrace.map((p) => p.policy_loss),
      },
      {
        label: "Value MSE",
        color: "#ff9a3c",
        points: trainTrace.map((p) => p.value_loss),
      },
    ],
    [trainTrace],
  );

  const gradSeries = useMemo(
    () => [
      {
        label: "Grad norm ‖∇‖",
        color: "#c4a574",
        points: trainTrace.map((p) => p.grad_norm),
      },
    ],
    [trainTrace],
  );

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

      <section className="selfplay-live" aria-label="Live self-play board">
        <div className="selfplay-live-board">
          <div className="selfplay-live-meta">
            <span>
              {liveMeta.phase === "training"
                ? "Training weights"
                : liveMeta.phase === "playing"
                  ? "Playing"
                  : "Ready"}
            </span>
            <span>
              {liveMeta.game > 0
                ? `Game ${liveMeta.game}/${liveMeta.games || "—"} · ${shortModel(liveMeta.model)}`
                : shortModel(model)}
            </span>
            <span>
              {liveMeta.rootValue != null
                ? `Root ${formatEval(liveMeta.rootValue)} (−1…+1)`
                : liveMoves.length
                  ? `Ply ${liveMoves.length}`
                  : "Waiting for ply"}
            </span>
          </div>
          <ChessBoard fen={liveFen} lastUci={lastUci} interactive={false} view3d={false} />
        </div>
        <div className="selfplay-live-moves panel">
          <h3>Move log</h3>
          <p className="muted-note">
            Each ply streams as MCTS finishes — the board advances one move at a time.
          </p>
          <dl className="selfplay-glossary">
            <div>
              <dt>SAN</dt>
              <dd>
                Standard Algebraic Notation — the human-readable move (e.g. <code>Nf3</code>,{" "}
                <code>exd5</code>).
              </dd>
            </div>
            <div>
              <dt>UCI</dt>
              <dd>
                Engine coordinate form — from-square + to-square (e.g. <code>g1f3</code>).
              </dd>
            </div>
            <div>
              <dt>Root</dt>
              <dd>
                MCTS-backed eval for the side to move (−1…+1). Near 0 ≈ drawish / value head not
                decisive yet — early training often sits around ±0.001, which used to display as
                0.00.
              </dd>
            </div>
            <div>
              <dt>Net</dt>
              <dd>Raw network value on this position before search backups.</dd>
            </div>
            <div>
              <dt>Move Q</dt>
              <dd>Search Q of the chosen move, flipped to the side-to-move’s view.</dd>
            </div>
          </dl>
          <div className="selfplay-move-head" aria-hidden="true">
            <span>#</span>
            <span>SAN</span>
            <span>UCI</span>
            <span>Root</span>
            <span>Net</span>
            <span>Move Q</span>
          </div>
          <div className="selfplay-move-list" ref={moveListRef}>
            {liveMoves.length === 0 ? (
              <p className="muted-note">No plies yet.</p>
            ) : (
              liveMoves.map((m) => (
                <div key={m.ply} className={`selfplay-move-row ${m.side}`}>
                  <span className="ply">{m.ply}.</span>
                  <span className="san" title="SAN — algebraic move">
                    {m.san}
                  </span>
                  <span className="uci" title="UCI — from/to squares">
                    {m.uci}
                  </span>
                  <span className="val" title="Root value — MCTS eval (−1…+1)">
                    {formatEval(m.root_value)}
                  </span>
                  <span className="val" title="Net value — network before search">
                    {formatEval(m.net_value)}
                  </span>
                  <span className="val" title="Move Q — chosen move from side-to-move view">
                    {formatEval(m.move_q)}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </section>

      <section className="selfplay-weights panel" aria-label="Weight update graphs">
        <h3>How weights update</h3>
        <p className="muted-note">
          After each self-play game, Adam runs on every trainable tensor: loss → backward → clip
          gradients → step. Graphs below cover that full pass (all layers / all weights), not a
          single head.
        </p>
        {weightMeta ? (
          <p className="muted-note">
            Last pass · game {weightMeta.game} · {weightMeta.samples} samples · {weightMeta.epochs}{" "}
            epochs · {weightMeta.steps} steps · lr {weightMeta.lr}
          </p>
        ) : (
          <p className="muted-note">Start self-play to stream train steps and per-layer Δw graphs.</p>
        )}
        {(trainTrace.length > 0 || layerDeltas.length > 0) && (
          <div className="selfplay-weight-charts">
            {trainTrace.length > 0 && (
              <>
                <LossChart
                  title="Train step loss (this game)"
                  series={trainLossSeries}
                  xLabel="Adam step →"
                />
                <LossChart title="Gradient norm per step" series={gradSeries} xLabel="Adam step →" />
              </>
            )}
            {layerDeltas.length > 0 && <LayerDeltaChart layers={layerDeltas} />}
            {weightHist && weightHist.counts.length > 0 && <WeightHistChart hist={weightHist} />}
          </div>
        )}
      </section>

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
