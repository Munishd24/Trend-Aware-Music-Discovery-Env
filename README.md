---
title: Music Discovery Env
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🎧 Trend-Aware Music Discovery Environment

> **Meta PyTorch OpenEnv Hackathon 2026** — A fully compliant OpenEnv RL benchmark where LLM agents act as intelligent music recommenders, optimizing for real user engagement signals across viral cultural moments.

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

**Live Demo:** [🚀 Open the Dashboard](https://munish24-music-discovery-env.hf.space/)

---

## 📖 What Is This?

This environment simulates an **AI-powered music recommendation engine** — the kind that powers real streaming platforms like Spotify. An LLM agent observes a listener's taste profile and a live catalog of trending songs, then must recommend the right track to maximize engagement.

The challenge is non-trivial: song trends decay over time, the user's mood is **hidden** (POMDP), and the global musical zeitgeist shifts every episode. The agent must balance exploitation (safe genre matches) against exploration (discovering new genres the user might love) — all within a strict 10-step budget.

This is **sequential user retention optimization** — a real, high-value business problem for any content platform.

---

## ✨ Key Features

### 1. Real Spotify Dataset (1,000 Tracks)
Powered by the [HuggingFace Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset). A pre-processing pipeline (`scripts/build_catalog.py`) filters for popularity > 60, maps audio features (energy, valence) to simulation variables, and generates a lightweight `server/catalog.json` for fast, latency-safe episode resets.

### 2. Hidden Mood State (POMDP)
The user's current mood is **never revealed** in the observation. Agents must infer it by reading the behavioral signal in `last_3_reactions`. Recommending a song that matches the hidden mood earns a bonus. This makes the task a true Partially Observable Markov Decision Process.

### 3. Global Viral Mood (Serendipity Multiplier)
Each episode begins with a randomly sampled `global_mood_trend` (e.g., `"party"`, `"hyped"`, `"sad"`). If an agent correctly recommends a song whose vibe matches **both** the user's hidden mood AND the global trend simultaneously, they unlock a massive `+1.0` serendipity retention multiplier. This simulates real-world viral discovery loops.

### 4. Success-Gated Diversity Bonus (Anti-Reward-Hacking)
Agents receive an `exploration_budget` of 2 attempts per episode to try a song from a **new genre** outside the user's known taste. The `+1.0` diversity bonus is awarded **only if** the user reacts positively (shared, saved, or added to playlist). A negative reaction wastes the attempt with no reward — preventing random genre-guessing exploits.

### 5. Premium Interactive Dashboard
A zero-click, auto-loading Gradio UI built in a Spotify-inspired dark glassmorphism style. On page load, the environment initializes instantly — judges see a live populated state without clicking anything. The activity history shows 🔥 badges for shares and 💎 indicators for serendipity combos.

### 6. Robust Autograder Inference Script
The `inference.py` script runs the LLM agent (Qwen2.5-72B-Instruct by default) across all 3 tasks with a strict 10-second per-call timeout. LLM responses are parsed via a 3-tier fallback chain (JSON → Regex → Heuristic Baseline), ensuring `success=true` even under API failure. Logs follow the mandatory `[START]`/`[STEP]`/`[END]` format exactly.

---

## 🏗️ Project Structure

```
RL/
├── inference.py                         # 🔑 LLM evaluation script (hackathon entry point)
├── openenv.yaml                         # OpenEnv spec: tasks, schemas, entrypoint
├── Dockerfile                           # Container definition (python:3.11-slim)
├── requirements.txt                     # Minimal dependencies
├── README.md                            # This file
│
├── models.py                            # Pydantic Action + Observation types
├── client.py                            # Typed WebSocket client
├── gradio_ui.py                         # Spotify-inspired Gradio dashboard
│
├── server/
│   ├── app.py                           # FastAPI app (OpenEnv create_app + custom endpoints)
│   ├── music_discovery_env_environment.py  # Core RL environment logic
│   └── catalog.json                     # Pre-built 1,000-song Spotify dataset
│
└── scripts/
    └── build_catalog.py                 # One-time script to regenerate catalog.json
```

---

## 🎮 Environment Design

### Observation — What the Agent Sees

```python
class MusicDiscoveryObservation(Observation):
    user: UserProfile           # Taste (genres, media_interests), discovery_openness
                                # ⚠️ user.mood is HIDDEN — must infer from last_3_reactions
    trending_songs: List[Song]  # Catalog with trend_velocity, trend_age_days, vibe, genre
    step_count: int             # Current step (max 10)
    session_engagement: List    # Full reaction log this episode
    recommended_history: List   # IDs already recommended (repeats are penalized)
    last_3_reactions: List[str] # Only signal for inferring the hidden mood
    global_mood_trend: str      # Current viral cultural trend (e.g., "party")
    session_genres: List[str]   # Genres explored so far this episode
    exploration_budget: int     # Remaining attempts for the +1.0 diversity bonus
```

### Action — What the Agent Returns

```python
class MusicDiscoveryAction(Action):
    song_id: str   # Must be a valid ID from trending_songs, not in recommended_history
```

### Reward Function

```
# Step reward is clamped to [-1.0, 1.0] to satisfy hackathon bounds
final_reward = max(-1.0, min(1.0, raw_reward))

raw_reward = (base_reaction × trend_freshness_multiplier) + taste_bonus + diversity_bonus

base_reaction:
  shared / saved    → +1.0 / +0.8
  added_to_playlist → +0.7
  played_once       → +0.3
  skipped / ignored → -0.2 / -0.3

trend_freshness_multiplier = max(0.5, 1.0 − trend_age_days × 0.05)
  (songs trending for 10+ days decay to a 0.5x multiplier)

taste_bonus:
  +0.2  if genre matches user's known taste_profile.genres
  +0.2  if song vibe matches hidden user mood (infer from reactions!)
  +1.0  [SERENDIPITY] if vibe matches BOTH hidden mood AND global_mood_trend

diversity_bonus:
  +1.0  if genre is NEW to session_genres AND reaction is highly positive
        (costs 1 exploration_budget — wasted on skip/played_once)
```

### Grading Function

```
episode_score = (engagement_rate × 0.4) + (avg_reward × 0.4) + (discovery_bonus × 0.2)

engagement_rate  = steps with positive reward / total steps
avg_reward       = mean reward across all steps
discovery_bonus  = steps where trend_age_days < 3 AND reaction in {shared, saved}
```

---

## 🎯 Task Difficulties

| Difficulty | Scenario | Key Challenge | Score Target |
|---|---|---|---|
| **Easy** | User loves anime. 10 fresh songs (< 3 days old). | Basic genre + trend matching | 0.80+ |
| **Medium** | Mixed catalog aging out. Mood shifts 25% chance/step. | POMDP mood inference under uncertainty | 0.45–0.65 |
| **Hard** | Cold-start user with no genre history. 20 diverse songs. | Exploration from scratch in 10 steps | 0.20–0.40 |

---

## 🚀 Running the Project

### Option 1: HuggingFace Space (Live Now)
Open the dashboard directly: **https://munish24-music-discovery-env.hf.space/**

### Option 2: Docker (Local)
```bash
docker build -t music-discovery-env .
docker run -p 7860:7860 -e ENABLE_WEB_INTERFACE=true music-discovery-env
# Open http://localhost:7860/
```

### Option 3: Local Development
```bash
source .venv/bin/activate
ENABLE_WEB_INTERFACE=true uvicorn server.app:app --port 7860 --reload
```

### Option 4: Run the Inference Script
```bash
# Against HuggingFace LLM (requires HF token):
HF_TOKEN=hf_... python3 inference.py

# Fallback heuristic mode (no token needed):
HF_TOKEN=dummy python3 inference.py
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset?task={easy\|medium\|hard}` | Start a new episode |
| `POST` | `/step` | Submit `{"action": {"song_id": "..."}}` |
| `GET` | `/state` | Get current observation |
| `WS` | `/ws` | WebSocket session (used by typed client) |
| `GET` | `/health` | Health check — returns `200 OK` |
| `GET` | `/` | **Spotify-style** interactive dashboard (auto-loads) |
| `GET` | `/web` | OpenEnv default Gradio playground |
| `GET` | `/docs` | Swagger / OpenAPI docs |
| `GET` | `/tasks` | Lists all 3 task configurations |
| `POST` | `/grader` | Grade a trajectory → `{"score": 0.85}` |
| `GET` | `/baseline` | Run deterministic baseline → `{"easy": ..., "medium": ..., "hard": ...}` |

---

## 🤖 Baseline Agent

The `baseline_agent()` in `server/music_discovery_env_environment.py` runs a heuristic strategy:

1. If `exploration_budget > 0`, 20% chance (ε-greedy) to pick the highest-velocity song from a **genre not yet played** — targeting the diversity bonus
2. Otherwise: score each unplayed song by genre match, global mood alignment, trend velocity, and recency
3. Always pick the top scorer

Scores **well on Easy** (clear signals), partially on **Medium** (POMDP inference is approximate), and struggles on **Hard** (cold start) — creating the clean difficulty gradient needed for evaluation.

---

## ✅ OpenEnv Compliance Checklist

- [x] `openenv.yaml` with `spec_version: 1`, `action_schema`, `observation_schema`, all 3 tasks
- [x] `Environment` subclass implementing `reset()`, `step()`, `state` property
- [x] `Action` and `Observation` Pydantic models with proper base class inheritance
- [x] `EnvClient` subclass for typed WebSocket communication
- [x] App created via `create_app()` — auto-generates `/reset`, `/step`, `/state`, `/ws`, `/health`
- [x] Custom `/grader` and `/baseline` endpoints
- [x] Root-level `Dockerfile` with `ENABLE_WEB_INTERFACE=true`
- [x] `inference.py` in root with `[START]`/`[STEP]`/`[END]` logging, 10s LLM timeout
- [x] `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME` environment variables
- [x] Rewards strictly clamped to `[-1.0, 1.0]`
- [x] HuggingFace Spaces metadata in `README.md` YAML frontmatter

---

## 📦 Dependencies

```
openenv-core>=0.2.2   # Core OpenEnv framework (auto-generates API)
fastapi>=0.115.0      # Web framework
uvicorn>=0.24.0       # ASGI server
pydantic              # Typed data models
openai                # LLM inference client
gradio                # Interactive web dashboard
```

---

## 📜 Acknowledgements

- Built for the **Meta PyTorch OpenEnv Hackathon 2026** (Round 1: March 25 – April 8)
- Framework: [OpenEnv / HuggingFace](https://github.com/huggingface/openenv-course)
- Dataset: [maharshipandya/spotify-tracks-dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)
- Research inspiration: Spotify's Sequential Playlist Generation & KDD 2023 RL Recommendation work
