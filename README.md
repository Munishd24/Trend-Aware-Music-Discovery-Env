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

> **Meta PyTorch OpenEnv Hackathon 2026** — An OpenEnv-compliant RL benchmark where LLM agents act as algorithmic music recommenders, optimizing engagement across viral cultural moments.

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📖 Overview

The **Trend-Aware Music Discovery Environment** simulates a real-world music recommendation engine. An LLM agent observes a user's taste profile, session history, and a catalogue of trending songs (each tied to viral cultural moments — anime, games, movies, TikTok). At each step, the agent must pick the optimal song to maximize user engagement before the cultural trend decays.

The benchmark is designed around three authentic RL problems: **echo-chamber retrieval**, **POMDP mood inference**, and **cold-start exploration**. It is fully compliant with the [OpenEnv](https://github.com/huggingface/openenv-course) evaluation standard.

---

## 🏗️ Architecture: Official OpenEnv 3-Component Pattern

The project strictly follows the official OpenEnv framework layout:

```
RL/
├── Dockerfile                           # Root-level container definition
├── README.md                            # This file (also HF Spaces metadata)
├── openenv.yaml                         # OpenEnv spec: tasks, schemas, entrypoint
├── requirements.txt                     # Python deps (openenv-core, gradio, openai…)
│
├── models.py                            # Action + Observation Pydantic types
├── client.py                            # Typed WebSocket client (MusicDiscoveryEnvClient)
├── __init__.py                          # Package exports
│
├── server/
│   ├── __init__.py
│   ├── app.py                           # FastAPI app via create_app() + custom endpoints
│   └── music_discovery_env_environment.py  # Core RL logic (Environment subclass)
│
├── gradio_ui.py                         # Spotify-inspired Gradio Blocks UI
├── inference.py                         # LLM agent evaluation script
├── music_discovery_proto_v2.py          # Legacy prototype (reference only)
└── main.py                              # Legacy standalone server (reference only)
```

### Component Roles

| File | Role |
|---|---|
| `models.py` | Defines `MusicDiscoveryAction` (inherits `Action`) and `MusicDiscoveryObservation` (inherits `Observation`) — the typed contract between agent and environment |
| `server/music_discovery_env_environment.py` | The core RL environment — inherits from `openenv.core.env_server.interfaces.Environment`, implements `reset()`, `step()`, and `state` |
| `server/app.py` | Creates the FastAPI app via `create_app()` which auto-generates `/reset`, `/step`, `/ws`, `/health`, `/web`. Custom `/tasks`, `/grader`, `/baseline` added on top |
| `client.py` | `MusicDiscoveryEnvClient` — typed WebSocket client used by `inference.py` |
| `gradio_ui.py` | The custom "Spotify-style" dashboard for manual demonstration |
| `inference.py` | LLM agent script orchestrating the full eval loop across all 3 tasks |

---

## 🎮 Environment Design

### Observation (what the agent sees)

```python
class MusicDiscoveryObservation(Observation):
    user: UserProfile           # Taste (genres, media_interests), discovery_openness
                                # NOTE: user.mood is HIDDEN (POMDP — infer from reactions)
    trending_songs: List[Song]  # Catalogue with trend_velocity, trend_age_days, vibe…
    step_count: int             # Current step (max 10)
    session_engagement: List    # Full reaction history this episode
    recommended_history: List   # IDs already recommended (never repeat)
    last_3_reactions: List[str] # Last 3 reactions — the only mood signal available
    global_mood_trend: str      # Dynamic cultural zeitgeist modifier (e.g., "party")
    session_genres: List[str]   # Genres played this episode (for diversity tracking)
    exploration_budget: int     # Remaining attempts to trigger the +1.0 diversity bonus
```

### Action (what the agent returns)

```python
class MusicDiscoveryAction(Action):
    song_id: str   # e.g. "song_03" — must be from trending_songs, not in recommended_history
```

### Reward Function

```
# Capped to strictly obey hackathon 0.0-1.0 evaluation bounds
final_reward = max(-1.0, min(1.0, raw_reward))

raw_reward = (base_reaction × trend_freshness_multiplier) + taste_bonus + diversity_bonus

base_reaction:
  shared / saved    → +1.0 / +0.8
  added_to_playlist → +0.7
  played_once       → +0.3
  skipped / ignored → -0.2 / -0.3

trend_freshness_multiplier = max(0.5, 1.0 − trend_age_days × 0.05)

taste_bonus:
  +0.2 if genre matches user's taste_profile.genres
  +0.2 if song vibe matches hidden user mood
  +1.0 [SERENDIPITY] if song vibe matches BOTH hidden mood AND global_mood_trend

diversity_bonus:
  +1.0 if genre is NEW to session_genres AND user reaction is highly positive (uses 1 exploration_budget)
```

### Grading Function (`/grader`)

```
score = (engagement_rate × 0.4) + (avg_reward × 0.4) + (discovery_bonus × 0.2)

discovery_bonus = steps where trend_age_days < 3 AND reaction in {shared, saved}
```

---

## 🎯 Task Difficulties

### Easy — The Echo Chamber
- **Scenario:** User has strong anime preferences. Catalog flooded with fresh (<3 days old) anime songs.
- **RL Test:** Basic content filtering and genre matching.
- **Noise level:** `±0.3` (highly predictable)
- **Baseline target:** `0.80+`

### Medium — Shifting Context + POMDP
- **Scenario:** Mixed catalog with aging trends. User mood shifts randomly (**25% chance per step**). Mood is **hidden** — agent must infer from `last_3_reactions`.
- **RL Test:** Temporal reasoning and context adaptation from partial information.
- **Noise level:** `±0.8`
- **Baseline target:** `0.45–0.65`

### Hard — Cold Start
- **Scenario:** New user with **no genre preferences**. Large diverse catalog (20 songs). Varied trend ages.
- **RL Test:** Classic exploration vs. exploitation — probe with universally viral songs to discover user taste within 10 steps.
- **Noise level:** `±0.5` (hard due to cold start, not noise)
- **Baseline target:** `0.20–0.35`

---

## ✨ Key Features & Technical Highlights

This environment was purpose-built to evaluate state-of-the-art LLMs against authentic recommendation challenges, eliminating reward-hacking vectors through strict protocols.

- **Real Spotify Dataset Engine:** Powered by a high-performance 1,000-track JSON catalog derived from official Hugging Face user datasets, mapping real track metadata (energy, valence) to dynamic simulation variables.
- **Hidden User Moods (POMDP):** The target user's mood is entirely hidden from the observation schema and shifts randomly. Agents are forced to actively reason and infer states strictly through trailing interaction logs (`last_3_reactions`).
- **Global Viral Thresholds (Serendipity):** Introduces a dynamic `global_mood_trend` representing wider cultural velocity. If an LLM successfully recommends a track that aligns with both the user's hidden POMDP mood AND the global viral trend, it acquires a massive retention multiplier!
- **Success-Gated Diversity (Exploration Budget):** Features an advanced anti-reward-hacking protocol. Agents are strictly bound by a multi-attempt `exploration_budget` to test new unseen genres. A massive `+1.0` bounty is awarded *only* if the user positively engages with the genre leap; otherwise, the attempt is wasted.
- **Premium Interactive Dashboard:** Features a stunning, zero-click auto-loading Gradio Blocks interface modeled after Spotify. The UI organically visualizes the environment state via `💎` Serendipity logging, `🔥` share trackers, formatted catalog outputs, and strict POMDP warning modules limit visibility.
- **Isolated, Robust Autograder (`inference.py`):** Ships with a fully autonomous evaluation script ensuring absolute compliance with Hackathon standards. Validates LLM responses via a robust failure cascade (JSON parse -> Regex -> Epsilon-Greedy Baseline) perfectly rendering `[START]`, `[STEP]`, and `[END]` evaluation logs.

---

## 🚀 Running the Project

### Option 1: Docker (Recommended for HF Spaces)

```bash
cd /home/munish/Projects/RL

# Build
docker build -t music-discovery-env .
   
# Run
docker run -d --name music-env -p 7860:7860 music-discovery-env

# Open Spotify-style web interface
open http://localhost:7860/
```

The Dockerfile sets `ENV ENABLE_WEB_INTERFACE=true` which activates the custom Gradio playground at the root `/` path.

### Option 2: Local uvicorn

```bash
cd /home/munish/Projects/RL
source .venv/bin/activate
uvicorn server.app:app --port 7860 --reload
```

### Option 3: Run inference (LLM agent)

```bash
# With real OpenAI key (LLM agent):
OPENAI_API_KEY=sk-... python3 inference.py

# Without key (heuristic fallback — still works):
python3 inference.py

# Against HF Space instead of local:
ENV_URL=https://your-username-music-discovery-env.hf.space python3 inference.py
```

The inference script auto-falls back to the heuristic agent if the LLM API call fails.

---

## 📡 API Reference

All endpoints auto-generated by `create_app()` unless marked **[Custom]**:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset?task={easy\|medium\|hard}` | Start a new episode |
| `POST` | `/step` | Submit action `{"action": {"song_id": "song_01"}}` |
| `GET` | `/state` | Current observation state |
| `WS` | `/ws` | WebSocket (used by `MusicDiscoveryEnvClient`) |
| `GET` | `/health` | Health check |
| `GET` | `/` | **[Custom]** Spotify-style interactive dashboard (auto-loads on page open) |
| `GET` | `/web` | OpenEnv default Gradio playground |
| `GET` | `/docs` | Swagger OpenAPI docs |
| `GET` | `/tasks` | **[Custom]** List easy/medium/hard configs |
| `POST` | `/grader` | **[Custom]** Grade trajectory `{"trajectory": [...]}` → `{"score": 0.85}` |
| `GET` | `/baseline` | **[Custom]** Run deterministic baseline → `{"easy": 0.92, "medium": 0.58, "hard": 0.24}` |

> ⚠️ **Note on Interfaces:** We provide two interfaces. `/web` is the standard OpenEnv auto-generated playground for developers. `/` (root) is our custom **Spotify-inspired** dashboard — it auto-loads the environment on page open so judges immediately see a live, populated state.

---

## 🤖 Baseline Agent

The `baseline_agent()` function in `server/music_discovery_env_environment.py` is a deterministic heuristic:
1. Filter songs to those matching user's genre preferences
2. From those, pick unplayed songs not in `recommended_history`
3. Among those, select the one with the highest `trend_velocity`

It scores well on **Easy** (predictable user, fresh songs) but collapses on **Hard** (no genre data to filter by). This creates the clear difficulty gradient needed for hackathon judging.

---

## 🔑 Key Design Decisions (Knowledge Transfer)

1. **Why `recommended_history` not `listening_history` for fallback?**
   The environment tracks `recommended_history` (all songs ever recommended this episode, even if repeated). The `listening_history` is the user-profile level list. The fallback and LLM prompt must use `recommended_history` to avoid the "stuck on same song" bug.

2. **Why WebSocket client in `inference.py`?**
   OpenEnv's `create_app()` maintains session state via WebSocket, not plain HTTP. The HTTP `/reset` + `/step` endpoints don't share session state between calls. The `MusicDiscoveryEnvClient` handles this transparently.

3. **Why is mood hidden?**
   To create a genuine POMDP. If mood were visible, the task would be trivial: just match `song.vibe == user.mood`. Hiding it forces the agent to reason from `last_3_reactions` to infer the hidden state.

4. **Why `/step` returns `{"observation": {...}, "reward": ..., "done": ...}`?**
   OpenEnv `create_app()` wraps all responses in this envelope. When writing raw `requests` calls (not using the client), always extract `.get("observation", ...)`.

5. **Why the Spotify-style UI?**
   To make the benchmark's "real world" utility immediately obvious. By mimicking a global streaming platform, judges can instantly recognize the roles of "Curated Playlists" (Observations), "Skipped Tracks" (Negative Rewards), and "Cultural Velocity" (Trending data).

---

## 📦 Dependencies

```
openenv-core[core]>=0.2.2   # Core OpenEnv framework
fastapi>=0.115.0              # Web framework (managed by openenv)
uvicorn>=0.24.0               # ASGI server
pydantic                      # Data validation
openai                        # LLM inference client
requests                      # HTTP calls in inference script
python-dotenv                 # .env file support
gradio                        # Web UI (activated by ENABLE_WEB_INTERFACE=true)
```

---

## 🌍 OpenEnv Compliance Checklist

- [x] `openenv.yaml` with `spec_version: 1`, `action_schema`, `observation_schema`, `tasks`
- [x] `Environment` subclass with `reset()`, `step()`, `state` property
- [x] `Action` and `Observation` base class inheritance in `models.py`
- [x] `EnvClient` subclass with `_step_payload`, `_parse_result`, `_parse_state`
- [x] App created via `create_app()` — not hand-rolled FastAPI
- [x] `/health`, `/ws`, `/web` auto-generated
- [x] Custom `/grader` and `/baseline` endpoints for hackathon evaluation
- [x] Root-level `Dockerfile` with `ENABLE_WEB_INTERFACE=true`
- [x] `requirements.txt` with pinned `openenv-core`
- [x] HF Spaces metadata in `README.md` YAML frontmatter

---

## 🔮 Future Work

- **Live Data:** Hook `_get_trending_songs()` into Spotify Trending API / Google Trends
- **RL Agents:** Benchmark PPO/DQN trained agents vs zero-shot LLM agents
- **Cross-Session Memory:** Evolve `MusicDiscoveryEnvironment` to persist and update `taste_profile` across episodes
- **Multi-User Simulation:** Run concurrent episodes with different user archetypes

---

## 📜 Citation & Acknowledgements

- Built for the **Meta PyTorch OpenEnv Hackathon 2026** (Round 1: March 25 – April 8)
- Framework: [OpenEnv Course / HuggingFace](https://github.com/huggingface/openenv-course)
- RL recommendation research inspiration: Spotify's sequential playlist generation work
