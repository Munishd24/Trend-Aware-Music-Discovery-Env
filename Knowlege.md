=============================================================
MUSIC DISCOVERY RL ENVIRONMENT — FULL PROJECT CONTEXT
Meta PyTorch OpenEnv Hackathon 2026
=============================================================
Generated: April 2026
Developer: Munish D (munishd.work@gmail.com)
HF Space: https://munish24-music-discovery-env.hf.space
Scaler Dashboard: https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard

=============================================================
SECTION 1 — WHAT THIS PROJECT IS
=============================================================

Project name: Trend-Aware Music Discovery Environment
Hackathon: Meta PyTorch OpenEnv Hackathon 2026 (Scaler SST x Meta x HuggingFace)
Round 1 deadline: April 7, 2026 11:59 PM
Submission: HF Space URL pasted into Scaler dashboard

Core concept:
An OpenEnv-compliant RL benchmark environment where an LLM agent
acts as a music recommender. The agent sees a user profile (taste,
mood inferred from reactions, media interests) and a catalogue of
trending songs tied to viral cultural moments (anime, games, movies,
TikTok). Each step the agent recommends one song. The environment
simulates whether the user engages based on taste match + trend
freshness. The agent learns to surface the right song to the right
user before the trend dies.

Real-world grounding:
Inspired directly by Spotify's KDD 2023 paper:
"Automatic Music Playlist Generation via Simulation-based
Reinforcement Learning" — Tomasi et al., Spotify, KDD 2023
Spotify identified that the critical missing component for applying
RL to music recommendation was an offline simulator. This environment
provides exactly that.

=============================================================
SECTION 2 — HACKATHON EVALUATION STRUCTURE
=============================================================

Phase 1 — Automated Validation (pass/fail gate):
- HF Space deploys and returns 200
- /reset returns valid observation
- /tasks returns task list
- /baseline runs without error and returns scores
- 3+ tasks with graders returning 0.0-1.0
- Dockerfile builds successfully
- openenv.yaml is valid

Phase 2 — Agentic Evaluation:
- Standard LLM (Nemotron-3-Super, 253B parameter NVIDIA model)
  runs against all submitted environments
- Score variance check across tasks
- Want to see clear difficulty progression

Phase 3 — Human Review:
- Meta and HuggingFace engineers review code
- Check: real-world utility, creativity, exploit resistance

Disqualification criteria:
- Environment does not deploy or respond
- Plagiarized or trivially modified existing environments
- Graders that always return same score
- No baseline inference script

=============================================================
SECTION 3 — FILE STRUCTURE
=============================================================

/home/munish/Projects/RL/
├── Dockerfile
├── README.md                    (HF Spaces metadata in YAML frontmatter)
├── openenv.yaml                 (OpenEnv spec)
├── requirements.txt
├── models.py                    (Pydantic models — Action + Observation)
├── client.py                    (MusicDiscoveryEnvClient — WebSocket)
├── __init__.py
├── server/
│   ├── __init__.py
│   ├── app.py                   (FastAPI via create_app())
│   └── music_discovery_env_environment.py  (Core RL logic)
├── inference.py                 (LLM agent evaluation script)
├── music_discovery_proto_v2.py  (Legacy prototype — reference only)
└── main.py                      (Legacy standalone — reference only)

=============================================================
SECTION 4 — ENVIRONMENT DESIGN
=============================================================

--- STATE SCHEMA ---

Observation (what agent sees):
{
  "user": {
    "taste_profile": {
      "genres": ["pop", "rock"],           # user's favourite genres
      "media_interests": ["anime"]          # anime/game/movie/tv_show/viral_video
    },
    "discovery_openness": 0.8,             # 0-1, willingness to try new things
    "listening_history": []                # songs played this session
  },
  "trending_songs": [                      # catalogue of available songs
    {
      "id": "song_01",
      "title": "Enemy",
      "artist": "Imagine Dragons",
      "source_media": "Arcane",
      "media_type": "tv_show",
      "trend_velocity": 0.9,               # 0-1, how fast trending
      "trend_age_days": 2,                 # days since went viral
      "genre": "pop",
      "vibe": "hyped"
    }
  ],
  "step_count": 0,                         # current step (max 10)
  "session_engagement": [],                # full reaction history
  "recommended_history": [],               # IDs already recommended (never repeat)
  "last_3_reactions": [],                  # ONLY mood signal available to agent
  "global_mood_trend": "party"             # Global viral trend modifier
}

IMPORTANT: user.mood is HIDDEN from observation (POMDP design)
Agent must infer mood from last_3_reactions only.

--- ACTION SCHEMA ---

{"song_id": "song_01"}

Must be from trending_songs, must NOT be in recommended_history.
If repeated: reward = -0.3, info = {"repeated": True}, episode continues.

--- REWARD FUNCTION ---

final_reward = base_reaction * trend_freshness_multiplier + taste_bonus

base_reaction values:
  shared            -> +1.0
  saved             -> +0.8
  added_to_playlist -> +0.7
  played_once       -> +0.3
  skipped           -> -0.2
  ignored (repeat)  -> -0.3

trend_freshness_multiplier = max(0.5, 1.0 - trend_age_days * 0.05)
  (songs older than 10 days get capped at 0.5 multiplier)

taste_bonus:
  +0.2 if song.genre in user.taste_profile.genres
  +0.2 if song.vibe matches hidden user.mood
  +1.0 (Serendipity) if song.vibe matches BOTH user.mood AND global_mood_trend
       *Also triggers +0.15 permanently to user's discovery_openness (Retention)*

--- GRADER FUNCTION ---

score = (engagement_rate * 0.4) + (avg_reward * 0.4) + (discovery_bonus * 0.2)

engagement_rate = positive_steps / total_steps
avg_reward = sum(rewards) / total_steps
discovery_bonus = steps where trend_age_days < 3 AND reaction in {shared, saved}

Returns float 0.0-1.0

=============================================================
SECTION 5 — THREE TASKS
=============================================================

EASY (difficulty 0.2):
- User: strong anime interest, genres [pop, rock], mood: hyped
- Catalogue: 10 songs, all from anime sources, trend_age 1-3 days
- Noise level: ±0.3 (highly predictable)
- Tests: basic content filtering and genre matching
- Baseline target: 0.80+
- RL concept: Echo chamber retrieval

MEDIUM (difficulty 0.5):
- User: mixed media interests, genres [rock, hip-hop]
- Catalogue: 15 songs, mixed sources, trend_age 3-15 days
- Mood shifts: 25% chance per step (random, not fixed every-3)
- Mood is HIDDEN — agent must infer from last_3_reactions
- Noise level: ±0.8
- Tests: temporal reasoning, POMDP inference, context adaptation
- Baseline target: 0.45-0.65
- RL concept: Partial observability (POMDP)

HARD (difficulty 0.8):
- User: cold start — no genre preferences, only 2 seed songs in history
- Catalogue: 20 songs (sampled from 100-song database), all media types
- Trend ages: 0-25 days (varies widely)
- discovery_openness: 0.3 (very low — picky user)
- Noise level: ±0.5 (hard due to cold start not noise)
- Tests: exploration vs exploitation, cold start inference
- Baseline target: 0.20-0.35
- RL concept: Cold start exploration

=============================================================
SECTION 6 — V3 HARDENING FEATURES
=============================================================

1. Partial Observability (POMDP)
   - user.mood removed from observation schema
   - Mood tracked internally only
   - last_3_reactions added as only mood signal
   - Forces agent to reason from behaviour, not explicit state

2. Stochastic Rewards
   - Probabilistic reaction thresholds, not hard cutoffs
   - Per-task noise bands: easy ±0.3, medium ±0.8, hard ±0.5
   - Perfect match song: minimum 65% chance of save/share
   - Poor match song: minimum 60% chance of skip
   - Mimics real-world human unpredictability

3. Random Mood Shifts (medium task only)
   - Changed from fixed every-3-steps to 25% chance per step
   - Uses MOOD_ROTATION = ["hyped", "relaxed", "party", "focus", "sad"]
   - Prevents gameable counting strategy

4. Repeat Penalty
   - Re-recommending any song: -0.3 reward, info={"repeated": True}
   - recommended_history tracked per episode
   - Prompt explicitly lists DO NOT REPEAT THESE songs

5. Seed Isolation
   - random.seed(42) in /baseline endpoint
   - Ensures identical scores every validator run
   - Required for automated validation

6. 100-Song Database
   - Expanded from 20 to 100 real cultural moment songs
   - Covers: anime, games, movies, TV shows, viral/TikTok
   - 10 distinct genres, 5 vibes
   - Prevents brute-force catalogue exhaustion

=============================================================
SECTION 6.1 — V4 DIFFERENTIATOR: MOOD-BASED TRENDS
=============================================================

1. Global Viral Moods
   - Overlays a dynamic "global_mood_trend" representing cultural zeitgeist.
   - Any song matching this trend generates a massive +0.5 boost to trend_velocity.

2. User Retention Serendipity Loops
   - If the agent recommends a track that hits BOTH the user's hidden POMDP mood AND the global viral mood, it triggers a serendipity event.
   - Reward multiplies heavily (+1.0 taste bonus).
   - Real-world retention metric simulation triggers: user's 'discovery_openness' is permanently increased (+0.15), making them highly receptive to broader genres going forward.

3. Premium Glassmorphism UI
   - Swapped raw standard Gradio for sleek, Spotify-inspired CSS aesthetics.
   - Implemented real-time dynamic overlay cards visualising the Viral Mood to hackathon judges.

=============================================================
SECTION 7 — REAL SONGS DATABASE (examples)
=============================================================

Key songs in database (all real cultural moments):
- song_01: Enemy - Imagine Dragons (Arcane, tv_show, pop, hyped)
- song_02: Chippin In - SAMURAI (Cyberpunk 2077, game, rock, hyped)
- song_03: SPECIALZ - King Gnu (Jujutsu Kaisen, anime, rock, hyped)
- song_04: Idol - YOASOBI (Oshi no Ko, anime, pop, party)
- song_05: Bling-Bang-Bang-Born - Creepy Nuts (Mashle, anime, hip-hop, party)
- song_06: Night Dancer - imase (TikTok, viral_video, pop, relaxed)
- song_07: Judas - Lady Gaga (JJK S2, anime, pop, dark)
- song_08: Running Up That Hill - Kate Bush (Stranger Things, tv_show, pop, sad)
- song_09: A Sky Full of Stars - Coldplay (Your Name movie, movie, pop, sad)
- song_10: Peaches - Jack Black (Super Mario Movie, movie, pop, party)
- song_11: Goth - Sidewalks and Skeletons (TikTok, viral_video, electronic, focus)
- song_12: Makeba - Jain (Levis Ad, tv_show, pop, party)
- song_13: Paint It Black - Rolling Stones (Wednesday, tv_show, rock, sad)
- song_14: Bloody Mary - Lady Gaga (Wednesday, tv_show, pop, party)
- song_15: Shikairo Days - Shika-bu (Nokotan, anime, pop, party)
- song_16: Suzume - RADWIMPS (Suzume movie, anime, pop, sad)
- song_17: The Last of Us - Gustavo Santaolalla (TLOU, game, classical, sad)
- song_18: Gurenge - LiSA (Demon Slayer, anime, rock, hyped)
- song_19: Unravel - TK (Tokyo Ghoul, anime, rock, sad)
- song_20: Megalovania - Toby Fox (Undertale, game, electronic, hyped)
+ 80 more songs across anime, games, movies, TV, viral categories

=============================================================
SECTION 8 — API ENDPOINTS
=============================================================

POST /reset?task={easy|medium|hard}
  Returns: full Observation JSON
  Example: curl -X POST "http://localhost:7860/reset?task=easy"

POST /step
  Body: {"action": {"song_id": "song_01"}}
  Returns: {observation, reward, done, info}
  NOTE: create_app() wraps action — use {"action": {"song_id": ...}}
  not flat {"song_id": ...}

GET /state
  Returns: current Observation

GET /tasks
  Returns: [{name, difficulty, description}] + action_schema

POST /grader
  Body: {"trajectory": [{step, song_id, reaction, reward, trend_age_days}]}
  Returns: {"score": 0.85}

GET /baseline
  Returns: {"easy": X, "medium": Y, "hard": Z}
  Uses random.seed(42) — deterministic

GET /health
  Returns: {"status": "ok"}

WS /ws
  WebSocket for MusicDiscoveryEnvClient

GET /web
  Gradio playground UI

GET /docs
  Swagger OpenAPI docs

=============================================================
SECTION 9 — INFRASTRUCTURE
=============================================================

--- DOCKERFILE ---

FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV ENABLE_WEB_INTERFACE=true
WORKDIR /app
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY --chown=user . /app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

Port: 7860 (required by HF Spaces)
Non-root user: required by HF Spaces
ENABLE_WEB_INTERFACE=true: activates Gradio UI

--- REQUIREMENTS ---

openenv-core[core]>=0.2.2
fastapi>=0.115.0
uvicorn>=0.24.0
pydantic
openai
requests
python-dotenv
gradio

--- HF SPACES README HEADER (must be at very top of README.md) ---

---
title: Music Discovery Env
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

--- GIT REMOTE ---

Remote name: space
URL: https://munish24:{HF_TOKEN}@huggingface.co/spaces/munish24/music-discovery-env
Push command: git push space master:main --force

IMPORTANT: Never commit .venv/ folder
.gitignore must include: .venv/, __pycache__/, *.pyc, *.so, *.pyo

=============================================================
SECTION 10 — INFERENCE SCRIPT DESIGN
=============================================================

File: inference.py (must be in root directory)
Runtime: under 20 minutes
Machine: 2 vCPU, 8GB RAM

Environment variables required:
  API_BASE_URL   = LLM API endpoint
  MODEL_NAME     = model identifier
  HF_TOKEN       = Strict requirement (No OPENAI_API_KEY allowed per rules)

Uses strictly OpenAI client initialized exclusively with HF_TOKEN pointed at API_BASE_URL.
Mandatory logging via standard [START], [STEP], and [END] structured JSON log hooks.

LLM prompt structure:
  SYSTEM: decision framework (5 rules), output format instruction
  USER: user profile section + DO NOT REPEAT section +
        available songs numbered list + task instruction +
        explicit list of valid song IDs

Robust JSON parsing chain:
  1. Try direct json.loads()
  2. If fails: regex search for {"song_id": "..."} pattern
  3. If fails: extract any song_XX pattern from response
  4. If all fail: fallback to highest trend_velocity unplayed song

Fallback is critical — Nemotron-3-Super outputs clean JSON but
smaller models hallucinate IDs like "EXCLUDE" or "ID_02".
Fallback prevents episode crashes.

--- SYSTEM PROMPT ---

You are an expert music recommendation agent optimizing for
user engagement and trend discovery.

Decision framework (apply in order):
1. EXCLUDE any song in recommended_history — never repeat
2. PRIORITIZE songs matching user's media_interests (strongest signal)
3. PREFER songs matching user's genre preferences
4. FAVOR songs with trend_age_days < 5 (fresher = higher reward)
5. Among equal options, pick highest trend_velocity

Output ONLY: {"song_id": "EXACT_ID_FROM_AVAILABLE_SONGS"}

=============================================================
SECTION 11 — BASELINE AGENT LOGIC
=============================================================

def baseline_agent(state):
    user = state["user"]
    genres = user["taste_profile"]["genres"]
    songs = state["trending_songs"]
    history = user.get("recommended_history", [])

    # Filter to genre matches
    matching = [s for s in songs if s["genre"] in genres] if genres else songs
    if not matching:
        matching = songs

    # Remove already recommended
    unplayed = [s for s in matching if s["id"] not in history]
    if not unplayed:
        unplayed = [s for s in songs if s["id"] not in history]
    if not unplayed:
        unplayed = songs

    # Pick highest trend_velocity
    best = max(unplayed, key=lambda s: s["trend_velocity"])
    return {"song_id": best["id"]}

This baseline:
- Works well on Easy (clear genre preferences, fresh songs)
- Struggles on Medium (mood shifts it can't detect)
- Collapses on Hard (no genre preferences to filter by)
Creates clear difficulty gradient.

=============================================================
SECTION 12 — BASELINE SCORES (TARGET)
=============================================================

After V3 hardening:
Easy:   0.80+ (was 0.89 before V3)
Medium: 0.45-0.65 (was 0.60 before V3)
Hard:   0.20-0.35 (was 0.30 before V3)

V3 actual scores reported: Easy 0.68, Medium 0.56, Hard 0.42
(Easy slightly low, Hard slightly high — calibration adjustment needed)

random.seed(42) ensures /baseline returns identical scores every run.

=============================================================
SECTION 13 — OPENENV COMPLIANCE CHECKLIST
=============================================================

[x] openenv.yaml with spec_version, action_schema, observation_schema, tasks
[x] Environment subclass with reset(), step(), state property
[x] Action and Observation base class inheritance in models.py
[x] EnvClient subclass (MusicDiscoveryEnvClient) with WebSocket
[x] App created via create_app() — not hand-rolled FastAPI
[x] /health, /ws, /web auto-generated by create_app()
[x] Custom /grader and /baseline endpoints
[x] Root-level Dockerfile with ENABLE_WEB_INTERFACE=true
[x] requirements.txt with pinned openenv-core
[x] HF Spaces metadata in README.md YAML frontmatter
[x] random.seed(42) in /baseline for reproducibility
[x] Repeat penalty with recommended_history tracking
[x] Robust JSON parsing with fallback chain in inference.py

=============================================================
SECTION 14 — KNOWN ISSUES AND FIXES
=============================================================

ISSUE: .venv folder committed to git
FIX: git filter-repo --path .venv --invert-paths --force
     Then git push space master:main --force

ISSUE: git push rejected (binary files)
FIX: Remove .venv from git history using git filter-repo
     Add .venv/ to .gitignore before first commit

ISSUE: git push rejected (fetch first)
FIX: git push space master:main --force

ISSUE: HF Space shows "Missing SDK in configuration"
FIX: Add sdk: docker to README.md YAML frontmatter

ISSUE: LLM hallucinating invalid song IDs
FIX: Add explicit valid ID list to end of user prompt
     Add robust parsing chain with fallback

ISSUE: Baseline scores vary between runs
FIX: Add random.seed(42) at start of baseline function

ISSUE: Agent recommends same song every step
FIX: Track recommended_history in state
     Penalize repeats with -0.3 reward
     Show DO NOT REPEAT THESE in LLM prompt

ISSUE: WebSocket vs HTTP session state
FIX: Use MusicDiscoveryEnvClient (WebSocket) not raw HTTP
     HTTP /reset + /step don't share session state

=============================================================
SECTION 15 — RESEARCH GROUNDING
=============================================================

Primary inspiration:
Spotify KDD 2023: "Automatic Music Playlist Generation via
Simulation-based Reinforcement Learning" — Tomasi et al.

Key parallel: Spotify needed an offline simulator before doing
anything else with RL. This environment IS that simulator,
open-sourced as a benchmark.

Spotify's production system uses:
- Trained neural network user model (on real data)
- AH-DQN (Action Head Deep Q-Network) agent
- TF-Agents framework
- Real Spotify interaction data

This environment uses:
- Probabilistic rule-based user simulation (no real data needed)
- LLM agent for evaluation
- OpenEnv framework
- Hardcoded real cultural moment songs

The gap (synthetic vs real data) is acknowledged in README
and positioned correctly as "open benchmark without
requiring proprietary data."

=============================================================
SECTION 16 — SUBMISSION CHECKLIST
=============================================================

Before submitting on Scaler dashboard:

[ ] HF Space is live and responding
    curl -X POST "https://munish24-music-discovery-env.hf.space/reset?task=easy"
    Must return valid JSON observation

[ ] All three tasks work
    curl -X POST ".../reset?task=medium"
    curl -X POST ".../reset?task=hard"

[ ] Baseline is reproducible (run twice, get same result)
    curl "https://munish24-music-discovery-env.hf.space/baseline"
    curl "https://munish24-music-discovery-env.hf.space/baseline"

[ ] Grader works
    curl -X POST ".../grader" -H "Content-Type: application/json"
    -d '{"trajectory": []}'

[ ] Tasks endpoint works
    curl "https://munish24-music-discovery-env.hf.space/tasks"

[ ] Pre-submission validator from Scaler dashboard passes
    (run against live HF Space URL)

[ ] inference.py runs without errors locally
    python3 inference.py

[ ] README has Spotify KDD 2023 citation

[ ] random.seed(42) confirmed in /baseline

Submission: Go to Scaler dashboard -> Submit Assessment
-> paste HF Space URL -> submit
Deadline: April 7, 2026 11:59 PM

=============================================================
SECTION 17 — WHAT MAKES THIS SUBMISSION STAND OUT
=============================================================

1. Novel problem domain
   Nobody else is building music/entertainment recommendation
   environments. Crowd will build DevOps, email, chatbots.

2. Academic grounding
   Directly parallels Spotify's KDD 2023 production system.
   Not a toy — a real research-backed benchmark.

3. V3 hardening makes it genuinely hard to exploit
   POMDP (hidden mood), stochastic rewards, random mood shifts,
   repeat penalty, 100-song catalogue.

4. Clean difficulty progression
   Easy/Medium/Hard have clear distinct RL concepts:
   echo chamber → POMDP → cold start exploration.

5. Real cultural moment songs
   Enemy from Arcane, Judas from JJK, Chippin In from Cyberpunk.
   Makes the benchmark immediately understandable to any judge.

6. Robust engineering
   WebSocket client, Gradio UI, Swagger docs, fallback chain
   in inference, seed isolation for reproducibility.

7. Deep "Retention" Simulator (Mood-Based Trends)
   Incorporates Spotify's latest KDD insights by simulating long-term metric retention hooks tied inherently to viral serendipity and global mood matching.

8. Premium Spotify-inspired Dashboard
   Custom HTML/CSS integration rendering a dark-mode glassmorphism interface, entirely overriding the standard, clunky default OpenEnv aesthetic.

=============================================================
END OF CONTEXT FILE
=============================================================