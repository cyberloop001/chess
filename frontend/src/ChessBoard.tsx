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

  return (
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
  );
}
