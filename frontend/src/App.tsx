import { useCallback, useEffect, useState } from "react";
import { AnalyticsPage } from "./AnalyticsPage";
import { ArenaPage } from "./ArenaPage";
import { HumanPage } from "./HumanPage";
import { fetchMatchHistory } from "./matchHistory";
import type { StoredMatch } from "./types";

type Page = "arena" | "human" | "analytics";

export default function App() {
  const [page, setPage] = useState<Page>("arena");
  const [matches, setMatches] = useState<StoredMatch[]>([]);
  const [duelLive, setDuelLive] = useState(false);
  const [humanLive, setHumanLive] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);

  const refreshHistory = useCallback(async () => {
    try {
      const next = await fetchMatchHistory();
      setMatches(next);
      setHistoryError(null);
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : "Failed to load analytics");
    }
  }, []);

  useEffect(() => {
    void refreshHistory();
  }, [refreshHistory]);

  const tagline =
    page === "arena"
      ? "Watch MLP+MCTS and Transformer+MCTS duel under a shared AlphaZero-style search."
      : page === "human"
        ? "Play against a model. After the game, it trains from the result."
        : "Self-play progress, loss improvement, and head-to-head duel analytics.";

  return (
    <div className="app">
      <header className="hero">
        <div className="hero-top">
          <h1 className="brand">Ply Arena</h1>
          <nav className="nav" aria-label="Primary">
            <button
              className={`nav-link ${page === "arena" ? "active" : ""}`}
              onClick={() => setPage("arena")}
            >
              Arena{duelLive ? " · live" : ""}
            </button>
            <button
              className={`nav-link ${page === "human" ? "active" : ""}`}
              onClick={() => setPage("human")}
            >
              Human{humanLive ? " · live" : ""}
            </button>
            <button
              className={`nav-link ${page === "analytics" ? "active" : ""}`}
              onClick={() => {
                void refreshHistory();
                setPage("analytics");
              }}
            >
              Analytics
            </button>
          </nav>
        </div>
        <p className="tagline">{tagline}</p>
        {historyError && page === "analytics" && (
          <p className="tagline">{historyError}</p>
        )}
      </header>

      <div className={page === "arena" ? "" : "page-hidden"} aria-hidden={page !== "arena"}>
        <ArenaPage onHistoryChange={() => void refreshHistory()} onRunningChange={setDuelLive} />
      </div>
      <div className={page === "human" ? "" : "page-hidden"} aria-hidden={page !== "human"}>
        <HumanPage onHistoryChange={() => void refreshHistory()} onRunningChange={setHumanLive} />
      </div>
      <div
        className={page === "analytics" ? "" : "page-hidden"}
        aria-hidden={page !== "analytics"}
      >
        <AnalyticsPage
          matches={matches}
          onClear={() => void refreshHistory()}
          onRefresh={() => void refreshHistory()}
        />
      </div>
    </div>
  );
}
