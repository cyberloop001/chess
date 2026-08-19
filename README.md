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
  training/     # self-play + trainer + replay
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

Open http://127.0.0.1:8080 — **Arena** (model vs model), **Human** (you vs a model), or **Analytics**.

## Deploy to GitHub Pages

GitHub Pages hosts the **React UI only**. MCTS, training, and WebSockets need a separate Python API (Render, Railway, Fly.io, a VPS, etc.).

### 1. Enable Pages on GitHub

1. Push this repo to GitHub (e.g. `cyberloop001/chess`).
2. **Settings → Pages → Build and deployment**
   - Source: **GitHub Actions** (not “Deploy from a branch”)
   - Save. Until this is set, `deploy-pages` fails with **404 Not Found**.
3. After enabling Pages, re-run **Actions → Deploy frontend to GitHub Pages** (or push again). The first deploy after enabling Pages should succeed.
4. Your site URL will be:
   - `https://<user>.github.io/<repo>/` (project site)
   - Example: `https://cyberloop001.github.io/chess/`

### 2. Deploy the API (required for Arena / Human / self-play)

Run FastAPI somewhere with HTTPS and WebSocket support, for example:

```bash
pip install -r requirements.txt
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

Copy `weights/*.pt` and optionally `data/` to that server if you want trained models.

CORS is already open (`allow_origins=["*"]`) so the Pages site can call the API.

### 3. Point the frontend at your API

In the GitHub repo: **Settings → Secrets and variables → Actions → Variables**

| Variable | Example |
|----------|---------|
| `VITE_API_URL` | `https://your-api.onrender.com` |

Rebuild by pushing to `main`/`dev` or re-running the Pages workflow. The build bakes this URL into the static JS.

Without `VITE_API_URL`, the UI loads on GitHub Pages but **duels and training will fail** (no backend on `github.io`).

### 4. Local production preview

```bash
cd frontend
npm run build
npm run preview
```

To mimic Pages + remote API:

```bash
VITE_API_URL=https://your-api.onrender.com npm run build
```

## Notes

- **Arena:** White = MLP, Black = Transformer. Set **Train count**, then each cycle is duel → train → next game.
- **Human:** Play as White or Black vs MLP or Transformer. Legal moves highlight on click; the opponent model trains after the game.
- After each Arena game, each net trains only on **its own moves** (MLP on White plies, Transformer on Black plies) plus its own replay file. Self-play trains one architecture against itself.
- **Analytics** are saved on the server in `data/match_history.json` (duels) and `data/selfplay_history.json` (self-play). The Analytics page charts self-play loss and improvement with citations.
- Self-play train (MLP or Transformer): games run until mate, a claimed draw (threefold / 50-move), resign, or 512 plies (then drawn). MCTS simulations default to 256, same as Arena. CLI: `python -m backend.training.self_play --model transformer --games 8`. You can also start a run from the Analytics dashboard.
