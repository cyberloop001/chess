const FILES = "abcdefgh";

const PIECE_NAMES: Record<string, string> = {
  K: "King",
  Q: "Queen",
  R: "Rook",
  B: "Bishop",
  N: "Knight",
};

function squareFromUci(uci: string, which: "from" | "to"): string | null {
  if (!uci || uci.length < 4) return null;
  const sq = which === "from" ? uci.slice(0, 2) : uci.slice(2, 4);
  const file = sq[0];
  const rank = sq[1];
  if (!FILES.includes(file) || rank < "1" || rank > "8") return null;
  return sq;
}

function promoName(letter: string): string {
  return PIECE_NAMES[letter.toUpperCase()] ?? letter.toUpperCase();
}

/** Plain-language explanation of SAN + UCI, e.g. Ke1 / c5 / Nxe5. */
export function explainSan(san: string, uci?: string): string {
  const raw = san.trim();
  if (!raw) return "Unknown move.";

  if (raw === "O-O" || raw === "0-0") {
    return "Castling kingside: king and rook swap to the short side.";
  }
  if (raw === "O-O-O" || raw === "0-0-0") {
    return "Castling queenside: king and rook swap to the long side.";
  }

  const checkmate = raw.includes("#");
  const check = !checkmate && raw.includes("+");
  const core = raw.replace(/[+#?!]/g, "");
  const suffix = checkmate ? " This is checkmate." : check ? " This gives check." : "";

  const from = uci ? squareFromUci(uci, "from") : null;
  const toUci = uci ? squareFromUci(uci, "to") : null;
  const promoUci = uci && uci.length >= 5 ? promoName(uci[4]) : null;

  if (core.includes("=")) {
    const [before, promoLetter] = core.split("=");
    const dest = before.replace(/x/g, "").slice(-2);
    const capture = before.includes("x");
    const promo = promoName(promoLetter);
    const origin = from ? ` from ${from}` : "";
    if (capture) {
      return `Pawn captures on ${dest}${origin} and promotes to a ${promo}.${suffix}`;
    }
    return `Pawn moves to ${dest}${origin} and promotes to a ${promo}.${suffix}`;
  }

  const isCapture = core.includes("x");
  const dest = toUci ?? core.replace(/x/g, "").slice(-2);
  const pieceLetter = /^[KQRBN]/.test(core) ? core[0] : "";
  const piece = pieceLetter ? PIECE_NAMES[pieceLetter] : "Pawn";
  const origin = from ? ` from ${from}` : "";

  if (promoUci && piece === "Pawn") {
    if (isCapture) {
      return `Pawn captures on ${dest}${origin} and promotes to a ${promoUci}.${suffix}`;
    }
    return `Pawn moves to ${dest}${origin} and promotes to a ${promoUci}.${suffix}`;
  }

  if (isCapture) {
    if (piece === "Pawn") {
      return `Pawn captures on ${dest}${origin}.${suffix}`;
    }
    return `${piece} captures on ${dest}${origin}.${suffix}`;
  }

  if (piece === "Pawn") {
    return `Pawn moves to ${dest}${origin ? origin : ""}.${suffix}`.replace(" .", ".");
  }
  return `${piece} moves to ${dest}${origin}.${suffix}`;
}

export function explainMoveTooltip(input: {
  ply: number;
  san: string;
  uci: string;
  side: "white" | "black";
  modelLabel: string;
}): { title: string; lines: string[] } {
  return {
    title: `Ply ${input.ply} · ${input.san}`,
    lines: [
      `${input.side === "white" ? "White" : "Black"} · ${input.modelLabel}`,
      `Played ${input.san} (${input.uci})`,
      explainSan(input.san, input.uci),
    ],
  };
}
