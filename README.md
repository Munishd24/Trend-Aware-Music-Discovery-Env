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

An OpenEnv-compliant RL environment where an LLM agent acts as a music recommender, optimizing for user engagement and trend freshness.

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg) ![License MIT](https://img.shields.io/badge/license-MIT-green.svg) ![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen.svg)

---

## 📖 Overview

The **Trend-Aware Music Discovery Environment** simulates a high-stakes, real-world recommendation engine where an LLM agent operates as a music discovery algorithm. At each step, the agent observes a user's static taste profile, transient mood, and historical engagement. It must select the optimal song from a catalog of trending tracks tied to viral cultural moments (e.g., gaming, anime, movies). The objective is to maximize user engagement by surfacing the right song at the exact right moment—before the cultural trend decays.

This benchmark grounds itself in a highly active area of machine learning research. Inspired by Spotify's ongoing research into Reinforcement Learning for user playlist recommendations, the environment reflects a reality where RL agents consistantly outperform standard collaborative filtering on sequential tasks. Every major streaming platform relies heavily on temporal recommendation ML at scale; however, this specific problem—assessing an LLM's capacity to simultaneously reason about taste, emotional state, and temporal trend decay—represents a novel domain currently unexplored in the OpenEnv ecosystem.

---

## ⚙️ Environment Design

### Observation Space
The state is represented as a JSON object capturing both the user context and the catalog of available trends:

* **`user`**: Object defining the listener's constraints.
  * `taste_profile`: Arrays of top `genres` and `media_interests`.
  * `mood`: The user's current emotional state (hyped, relaxed, sad, etc.).
  * `discovery_openness`: A float representing the user's willingness to step outside their known genres.
  * `listening_history`: Array of `song_id`s already consumed during the session.
* **`trending_songs`**: An array of candidate songs featuring metadata: `id`, `title`, `artist`, `source_media`, `media_type`, `trend_velocity` (0.0-1.0), `trend_age_days`, `genre`, and `vibe`.
* **`step_count`**: Integer representing the current phase of the episode (max 10).
* **`session_engagement`**: Historical array of user reactions in the current trajectory.

### Action Space
Agents must return a strictly formatted JSON object dictating the recommended track:
```json
{
  "song_id": "song_14"
}
```

### Reward Function
Rewards are calculated via a composite function incorporating the raw engagement metric, a taste-match bonus, and an early-discovery decay multiplier.

**Base Reaction Rewards:**
* `shared` → `+1.0`
* `saved` → `+0.8`
* `added_to_playlist` → `+0.7`
* `played_once` → `+0.3`
* `skipped` → `-0.2`

**Modifiers:**
1. **Taste Match Bonus:** `+0.2` if the song's genre matches the user's profile, and `+0.2` if the song's vibe matches the user's current mood.
2. **Early Discovery Bonus:** Rewards are multiplied by the freshness of the trend. Agents are penalized for recommending "dead" trends.
   * `Multiplier = max(0.5, 1.0 - (trend_age_days * 0.05))`

---

## 🎯 Tasks

| Task Name | Difficulty | Description | Baseline Score |
| :--- | :---: | :--- | :---: |
| **Easy** | `0.2` | User has a strong anime interest, and all candidate songs are sourced from fresh anime trends (1-3 days old). | ~0.89 |
| **Medium** | `0.5` | Features mixed media sources with aging trends (3-15 days old). The user's mood dynamically shifts every 3 steps. | ~0.60 |
| **Hard** | `0.8` | Absolute cold-start user constraint with only 2 seed songs. Features 20 diverse tracks across all media types with highly varied trend ages (0-20 days). | ~0.16 |

---

## 🚀 Quickstart

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Docker Build & Run**
The environment is fully containerized for Hugging Face Spaces deployment:
```bash
docker build -t music-discovery-env .
docker run -d -p 7860:7860 music-discovery-env
```

**3. Interact with the Environment**
Reset the environment to grab the initial state:
```bash
curl -X POST "http://localhost:7860/reset?task=medium"
```
Submit an action:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"song_id": "song_02"}' http://localhost:7860/step
```

---

## 📡 API Reference

| Method | Path | Description | Example Request / Response |
| :--- | :--- | :--- | :--- |
| `POST` | `/reset?task={easy\|medium\|hard}` | Resets environment and generates the task config. | **Res:** `{"user": {...}, "trending_songs": [...]}` |
| `POST` | `/step` | Submits an action to the environment. | **Req:** `{"song_id": "song_01"}`<br>**Res:** `StepResult` object with `reward`, `done`, `info`. |
| `GET` | `/state` | Returns the current observation state. | **Res:** `Observation` JSON object. |
| `GET` | `/tasks` | Lists task configurations and action schema. | **Res:** `{"tasks": [...], "action_schema": {...}}` |
| `POST` | `/grader` | Evaluates a completed trajectory. | **Req:** `{"trajectory": [...]}`<br>**Res:** `{"score": 0.85}` |
| `GET` | `/baseline` | Executes the baseline script across all tasks. | **Res:** `{"easy": 0.89, "medium": 0.60, "hard": 0.16}` |

---

## 🤖 Baseline Agent

Included in the environment is a deterministic `baseline_agent`. It evaluates the current `trending_songs` list and automatically selects the candidate with the highest recorded `trend_velocity` that matches the user's top genre. 

Because it lacks the reasoning capacity to optimize for *both* decaying trend age and shifting moods simultaneously, it scores remarkably well on `Easy` tasks, but severely struggles when met with complex, cold-start `Hard` tasks.

**Run the Baseline LLM Inference:**
*Note: A local `.venv` should be used to satisfy `PEP-668` restrictions.*
```bash
export OPENAI_API_KEY="your-api-key"
# (Or set API_BASE_URL to an OpenAI-compatible local model like Ollama)
python3 inference.py
```

---

## 📁 Project Structure

```text
.
├── Dockerfile                  # Container instructions
├── README.md                   # Environment documentation
├── inference.py                # LLM execution testing script
├── main.py                     # FastAPI wrapper and endpoints
├── models.py                   # Strict Pydantic interface types
├── music_discovery_proto_v2.py # Core environment logic and task generation
├── openenv.yaml                # Standard OpenEnv metadata schema
└── requirements.txt            # Python dependencies
```

---

## 🌍 Real-World Applications

Predictive curation is the lifeblood of the modern attention economy. Platforms like **Spotify, Netflix, Apple Music, and YouTube Music** rely entirely on sequential recommendation to maintain active users. 

Historically, collaborative filtering (CF) dominated this space. However, as demonstrated by Spotify's internal RL research, collaborative filtering fails gracefully when confronted with sequential nuances like temporal trend decay, abrupt mood shifts, and immediate cold-start environments. Reinforcement Learning actively outperforms standard CF paradigms by viewing recommendations as a multi-step trajectory rather than a static matrix completion problem.

---

## 🔮 Future Work

* **Live Data Integration:** Hooking the `_get_trending_songs()` generator into the Spotify Trending API and Google Trends for real-time validation.
* **RL Agent Benchmarking:** Evaluating standard PPO/DQN trained agents against zero-shot LLM conversational agents.
* **Cross-Session Memory:** Evolving the RL logic to dynamically learn and update the user's base `taste_profile` across long-term sequential episodes.

---

## 📜 Citation & Acknowledgements

* Built for the **Meta PyTorch OpenEnv Hackathon 2026**.
* Environment logic and scaling factors heavily inspired by **Spotify's RL recommendation research** for sequential playlist generation.
