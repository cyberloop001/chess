import { useCallback, useState } from "react";
import { AnalyticsPage } from "./AnalyticsPage";
import { ArenaPage } from "./ArenaPage";
import { loadMatchHistory } from "./matchHistory";
import type { StoredMatch } from "./types";

type Page = "arena" | "analytics";

export default function App() {
  const [page, setPage] = useState<Page>("arena");
  const [matches, setMatches] = useState<StoredMatch[]>(() => loadMatchHistory());
  const [duelLive, setDuelLive] = useState(false);

  const refreshHistory = useCallback(() => {
    setMatches(loadMatchHistory());
  }, []);

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
              className={`nav-link ${page === "analytics" ? "active" : ""}`}
              onClick={() => {
                refreshHistory();
                setPage("analytics");
              }}
            >
              Analytics
            </button>
          </nav>
        </div>
        <p className="tagline">
          {page === "arena"
            ? "Watch MLP+MCTS and Transformer+MCTS duel under a shared AlphaZero-style search."
            : "Compare model outcomes, search confidence, and value trajectories across duels."}
        </p>
      </header>

      {/* Keep both pages mounted so a running duel is not reset on tab switch */}
      <div className={page === "arena" ? "" : "page-hidden"} aria-hidden={page !== "arena"}>
        <ArenaPage onHistoryChange={refreshHistory} onRunningChange={setDuelLive} />
      </div>
      <div
        className={page === "analytics" ? "" : "page-hidden"}
        aria-hidden={page !== "analytics"}
      >
        <AnalyticsPage matches={matches} onClear={refreshHistory} />
      </div>
    </div>
  );
}
