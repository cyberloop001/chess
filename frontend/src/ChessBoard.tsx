import { Chess } from "chess.js";

const PIECES: Record<string, string> = {
  wK: "♔",
  wQ: "♕",
  wR: "♖",
  wB: "♗",
  wN: "♘",
  wP: "♙",
  bK: "♚",
  bQ: "♛",
  bR: "♜",
  bB: "♝",
  bN: "♞",
  bP: "♟",
};

const START_COUNTS: Record<"w" | "b", Record<string, number>> = {
  w: { p: 8, n: 2, b: 2, r: 2, q: 1, k: 1 },
  b: { p: 8, n: 2, b: 2, r: 2, q: 1, k: 1 },
};

const CAPTURE_ORDER = ["q", "r", "b", "n", "p"] as const;

type CapturedPieces = {
  white: string[];
  black: string[];
};

function capturedFromFen(fen: string): CapturedPieces {
  const chess = new Chess(fen);
  const onBoard: Record<"w" | "b", Record<string, number>> = {
    w: { p: 0, n: 0, b: 0, r: 0, q: 0, k: 0 },
    b: { p: 0, n: 0, b: 0, r: 0, q: 0, k: 0 },
  };
  for (const row of chess.board()) {
    for (const square of row) {
      if (!square) continue;
      onBoard[square.color][square.type] += 1;
    }
  }

  const white: string[] = [];
  const black: string[] = [];
  for (const type of CAPTURE_ORDER) {
    const missingWhite = Math.max(0, START_COUNTS.w[type] - onBoard.w[type]);
    const missingBlack = Math.max(0, START_COUNTS.b[type] - onBoard.b[type]);
    for (let i = 0; i < missingWhite; i += 1) white.push(`w${type.toUpperCase()}`);
    for (let i = 0; i < missingBlack; i += 1) black.push(`b${type.toUpperCase()}`);
  }
  return { white, black };
}

function CapturedTray({
  label,
  pieces,
  side,
}: {
  label: string;
  pieces: string[];
  side: "left" | "right";
}) {
  return (
    <aside className={`captured-tray captured-${side}`} aria-label={label}>
      <div className="captured-list">
        {pieces.length === 0 ? (
          <span className="captured-empty">—</span>
        ) : (
          pieces.map((key, i) => (
            <span key={`${key}-${i}`} className="captured-piece" data-color={key[0]}>
              {PIECES[key]}
            </span>
          ))
        )}
      </div>
    </aside>
  );
}

type Props = {
  fen: string;
  lastUci?: string | null;
  selected?: string | null;
  legalTargets?: string[];
  interactive?: boolean;
  orientation?: "white" | "black";
  onSquareClick?: (square: string) => void;
};

export function ChessBoard({
  fen,
  lastUci,
  selected = null,
  legalTargets = [],
  interactive = false,
  orientation = "white",
  onSquareClick,
}: Props) {
  const chess = new Chess(fen);
  const board = chess.board();
  const from = lastUci?.slice(0, 2);
  const to = lastUci?.slice(2, 4);
  const legalSet = new Set(legalTargets);
  const flipped = orientation === "black";

  const rows = flipped ? [...board].reverse() : board;
  const captured = capturedFromFen(fen);
  const leftPieces = flipped ? captured.white : captured.black;
  const rightPieces = flipped ? captured.black : captured.white;
  const leftLabel = flipped ? "White captured" : "Black captured";
  const rightLabel = flipped ? "Black captured" : "White captured";

  return (
    <div className="board-row">
      <CapturedTray label={leftLabel} pieces={leftPieces} side="left" />
      <div className={`board ${interactive ? "board-interactive" : ""}`} aria-label="Chess board">
      {rows.map((row, rankIdx) => {
        const displayRow = flipped ? [...row].reverse() : row;
        return displayRow.map((square, fileIdx) => {
          const rank = flipped ? rankIdx + 1 : 8 - rankIdx;
          const file = "abcdefgh"[flipped ? 7 - fileIdx : fileIdx];
          const sq = `${file}${rank}`;
          const dark = (rankIdx + fileIdx) % 2 === 1;
          const pieceKey = square ? `${square.color}${square.type.toUpperCase()}` : null;
          const isLegal = legalSet.has(sq);
          const isCapture = isLegal && Boolean(square);
          const classes = [
            "square",
            dark ? "dark" : "light",
            from === sq ? "last-from" : "",
            to === sq ? "last-to" : "",
            selected === sq ? "selected" : "",
            isLegal ? "legal" : "",
            isCapture ? "legal-capture" : "",
            interactive ? "clickable" : "",
          ]
            .filter(Boolean)
            .join(" ");

          return (
            <button
              key={sq}
              type="button"
              className={classes}
              disabled={!interactive}
              onClick={() => onSquareClick?.(sq)}
            >
              {(flipped ? rankIdx === 0 : rankIdx === 7) && (
                <span className="coord file">{file}</span>
              )}
              {(flipped ? fileIdx === 7 : fileIdx === 0) && (
                <span className="coord rank">{rank}</span>
              )}
              {pieceKey ? (
                <span className="piece" data-color={square?.color}>
                  {PIECES[pieceKey]}
                </span>
              ) : null}
              {isLegal && !isCapture ? <span className="legal-dot" /> : null}
              {isCapture ? <span className="legal-ring" /> : null}
            </button>
          );
        });
      })}
      </div>
      <CapturedTray label={rightLabel} pieces={rightPieces} side="right" />
    </div>
  );
}
