import { useEffect, useMemo, useState } from "react";
import {
  clearMatchHistoryRemote,
  computeAnalytics,
  type AnalyticsSummary,
} from "./matchHistory";
import { SelfPlayPanel } from "./SelfPlayPanel";
import { modelLabel, shortModel, type StoredMatch } from "./types";

type Props = {
  matches: StoredMatch[];
  onClear: () => void;
  onRefresh?: () => void;
};

function pct(n: number): string {
  return `${Math.round(n * 100)}%`;
}

function formatWhen(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function LineChart({
  title,
  series,
}: {
  title: string;
  series: Array<{ label: string; color: string; points: number[] }>;
}) {
  const width = 640;
  const height = 220;
  const pad = { t: 16, r: 16, b: 28, l: 40 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;
  const all = series.flatMap((s) => s.points);
  const minY = all.length ? Math.min(-1, ...all) : -1;
  const maxY = all.length ? Math.max(1, ...all) : 1;
  const span = maxY - minY || 1;

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
        <line
          x1={pad.l}
          x2={width - pad.r}
          y1={y(0)}
          y2={y(0)}
          className="chart-zero"
        />
        {[-1, -0.5, 0, 0.5, 1].map((tick) => (
          <g key={tick}>
            <text x={pad.l - 8} y={y(tick) + 4} className="chart-axis" textAnchor="end">
              {tick}
            </text>
          </g>
        ))}
        {series.map((s) => {
          if (s.points.length === 0) return null;
          const d = s.points
            .map((v, i) => `${i === 0 ? "M" : "L"} ${x(i, s.points.length)} ${y(v)}`)
            .join(" ");
          return <path key={s.label} d={d} fill="none" stroke={s.color} strokeWidth="2.5" />;
        })}
        <text x={pad.l} y={height - 8} className="chart-axis">
          Ply →
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

function BarCompare({
  title,
  leftLabel,
  rightLabel,
  left,
  right,
}: {
  title: string;
  leftLabel: string;
  rightLabel: string;
  left: number;
  right: number;
}) {
  const total = left + right || 1;
  const leftPct = (left / total) * 100;
  const rightPct = (right / total) * 100;
  return (
    <div className="chart-block">
      <h3>{title}</h3>
      <div className="bar-track" aria-label={title}>
        <div className="bar-left" style={{ width: `${leftPct}%` }} />
        <div className="bar-right" style={{ width: `${rightPct}%` }} />
      </div>
      <div className="bar-labels">
        <span>
          {leftLabel} · {left}
        </span>
        <span>
          {rightLabel} · {right}
        </span>
      </div>
    </div>
  );
}

function selectedValueSeries(match: StoredMatch | null) {
  if (!match) return [];
  const mlp: number[] = [];
  const transformer: number[] = [];
  for (const move of match.moves) {
    const key = move.model.toLowerCase();
    if (key.includes("mlp")) mlp.push(move.rootValue);
    if (key.includes("transformer")) transformer.push(move.rootValue);
  }
  return [
    { label: "MLP root value", color: "#ff9a3c", points: mlp },
    { label: "Transformer root value", color: "#7ec8ff", points: transformer },
  ];
}

function selectedConfidenceSeries(match: StoredMatch | null) {
  if (!match) return [];
  const mlp: number[] = [];
  const transformer: number[] = [];
  for (const move of match.moves) {
    const conf = move.topVisits / Math.max(1, move.totalTopVisits);
    const key = move.model.toLowerCase();
    if (key.includes("mlp")) mlp.push(conf);
    if (key.includes("transformer")) transformer.push(conf);
  }
  return [
    { label: "MLP top-visit share", color: "#ff9a3c", points: mlp },
    { label: "Transformer top-visit share", color: "#7ec8ff", points: transformer },
  ];
}

export function AnalyticsPage({ matches, onClear, onRefresh }: Props) {
  const summary: AnalyticsSummary = useMemo(() => computeAnalytics(matches), [matches]);
  const [selectedId, setSelectedId] = useState<string | null>(matches[0]?.id ?? null);
  const [busy, setBusy] = useState(false);
  const selected = matches.find((m) => m.id === selectedId) ?? matches[0] ?? null;

  useEffect(() => {
    if (matches.length > 0 && !matches.some((m) => m.id === selectedId)) {
      setSelectedId(matches[0]?.id ?? null);
    }
  }, [matches, selectedId]);

  return (
    <main className="analytics">
      <SelfPlayPanel />

      <section className="analytics-hero-row">
        <div>
          <h2 className="analytics-title">Duel analytics</h2>
          <p className="analytics-sub">
            Server-saved head-to-head results and MCTS search signals across finished duels.
          </p>
        </div>
        <div className="actions">
          <button className="btn btn-ghost" disabled={busy} onClick={() => onRefresh?.()}>
            Refresh
          </button>
          <button
            className="btn btn-ghost"
            disabled={matches.length === 0 || busy}
            onClick={() => {
              setBusy(true);
              void clearMatchHistoryRemote()
                .then(() => {
                  onClear();
                  setSelectedId(null);
                })
                .finally(() => setBusy(false));
            }}
          >
            Clear history
          </button>
        </div>
      </section>

      {summary.finishedMatches === 0 ? (
        <section className="panel analytics-empty">
          <h3>No finished matches yet</h3>
          <p>
            Run a full duel on the Arena page. Completed games are saved on the backend server at{" "}
            <code>data/match_history.json</code> and appear here.
          </p>
        </section>
      ) : (
        <>
          <section className="stat-grid">
            <article className="stat">
              <span className="stat-label">Finished games</span>
              <strong>{summary.finishedMatches}</strong>
            </article>
            <article className="stat">
              <span className="stat-label">Avg plies</span>
              <strong>{summary.avgPlies.toFixed(1)}</strong>
            </article>
            <article className="stat">
              <span className="stat-label">MLP win rate</span>
              <strong>{pct(summary.mlp.winRate)}</strong>
            </article>
            <article className="stat">
              <span className="stat-label">Transformer win rate</span>
              <strong>{pct(summary.transformer.winRate)}</strong>
            </article>
          </section>

          <section className="analytics-grid">
            <div className="panel">
              <BarCompare
                title="Head-to-head wins"
                leftLabel="MLP"
                rightLabel="Transformer"
                left={summary.headToHead.mlpWins}
                right={summary.headToHead.transformerWins}
              />
              <p className="muted-note">Draws: {summary.headToHead.draws}</p>
            </div>

            <div className="panel">
              <h3>Model cards</h3>
              <div className="model-cards">
                {[summary.mlp, summary.transformer].map((m) => (
                  <div key={m.id} className="model-card">
                    <strong>{modelLabel(m.id)}</strong>
                    <div>
                      {m.wins}W · {m.losses}L · {m.draws}D
                    </div>
                    <div>Avg root value {m.avgRootValue.toFixed(3)}</div>
                    <div>Avg game length {m.avgPly.toFixed(1)} plies</div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section className="analytics-grid analytics-grid-wide">
            <div className="panel">
              <h3>Recent matches</h3>
              <div className="match-table-wrap">
                <table className="match-table">
                  <thead>
                    <tr>
                      <th>When</th>
                      <th>Ivory</th>
                      <th>Sun</th>
                      <th>Result</th>
                      <th>Plies</th>
                      <th>Sims</th>
                    </tr>
                  </thead>
                  <tbody>
                    {summary.recent.map((m) => (
                      <tr
                        key={m.id}
                        className={selected?.id === m.id ? "selected" : ""}
                        onClick={() => setSelectedId(m.id)}
                      >
                        <td>{formatWhen(m.playedAt)}</td>
                        <td>{shortModel(String(m.white))}</td>
                        <td>{shortModel(String(m.black))}</td>
                        <td>{m.result}</td>
                        <td>{m.moves.length}</td>
                        <td>{m.simulations}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="panel">
              <LineChart title="Root value by ply (selected match)" series={selectedValueSeries(selected)} />
              <LineChart
                title="Search confidence by ply (top-visit share)"
                series={selectedConfidenceSeries(selected)}
              />
              {selected && (
                <p className="muted-note">
                  Selected: {shortModel(String(selected.white))} vs{" "}
                  {shortModel(String(selected.black))} · {selected.result} ·{" "}
                  {formatWhen(selected.playedAt)}
                </p>
              )}
            </div>
          </section>
        </>
      )}
    </main>
  );
}
