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
};

export function ChessBoard({ fen, lastUci }: Props) {
  const chess = new Chess(fen);
  const board = chess.board();
  const from = lastUci?.slice(0, 2);
  const to = lastUci?.slice(2, 4);

  return (
    <div className="board" aria-label="Chess board">
      {board.map((row, rankIdx) =>
        row.map((square, fileIdx) => {
          const rank = 8 - rankIdx;
          const file = "abcdefgh"[fileIdx];
          const sq = `${file}${rank}`;
          const dark = (rankIdx + fileIdx) % 2 === 1;
          const pieceKey = square ? `${square.color}${square.type.toUpperCase()}` : null;
          const classes = [
            "square",
            dark ? "dark" : "light",
            from === sq ? "last-from" : "",
            to === sq ? "last-to" : "",
          ]
            .filter(Boolean)
            .join(" ");

          return (
            <div key={sq} className={classes}>
              {rankIdx === 7 && <span className="coord file">{file}</span>}
              {fileIdx === 0 && <span className="coord rank">{rank}</span>}
              {pieceKey ? PIECES[pieceKey] : null}
            </div>
          );
        }),
      )}
    </div>
  );
}
