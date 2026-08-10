# Ply Arena
AlphaZero-style chess duel: **MLP + MCTS** vs **Transformer + MCTS**, with a live web UI.

## Architecture

Both agents share the same AlphaZero loop:

1. Encode the board as 19 feature planes (pieces, castling, EP, progress)
2. Network outputs a policy over 4672 AlphaZero actions + a value in `[-1, 1]`
3. PUCT MCTS uses the policy as priors and the value for leaf evaluation
4. The move with the highest visit count is played

| Agent | Network |
|-------|---------|
| `mlp` | Flattened board → dense trunk → policy/value heads |
| `transformer` | 64 square tokens → Transformer encoder → per-square policy planes + pooled value |

## Project layout

```
backend/
  chess_core/   # encoding + config
  models/       # MLP + Transformer
  mcts/         # shared PUCT search
  game/         # match orchestration
  api/          # FastAPI + WebSocket
  training/     # self-play stub
frontend/       # React match UI (Ply Arena)
weights/        # optional *.pt checkpoints (mlp.pt / transformer.pt)
```

## Setup

```bash
# from repo root
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

cd frontend
npm install
```

## Run

Terminal 1 — API:

```bash
python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2 — UI:

```bash
cd frontend
npm run dev
```

Open http://127.0.0.1:8080 and click **Start duel**.

## Notes

- **Arena:** White = MLP, Black = Transformer. One game per Start; shows win / loss / draw, then both nets self-train.
- **Analytics** are saved on the server in `data/match_history.json` (API: `/api/analytics/matches`).
- Checkpoints save to `weights/mlp.pt` / `weights/transformer.pt`.
- Self-play stub: `python -m backend.training.self_play --model mlp --simulations 64`
