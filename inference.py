"""
Inference script for the Trend-Aware Music Discovery RL Benchmark.

Connects to the OpenEnv server (local or HF Space) via the typed WebSocket
MusicDiscoveryEnvClient, runs an LLM agent over 3 task difficulties, and
grades each trajectory using the /grader endpoint.
"""

import os
import json
import re
import requests
from openai import OpenAI

# Typed OpenEnv WebSocket client
from client import MusicDiscoveryEnvClient
from models import MusicDiscoveryAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
# Some systems might pass standard OPENAI_API_KEY, we prefer HF_TOKEN if available
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", HF_TOKEN or "dummy_key")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

# Initialize OpenAI client according to hackathon instructions
openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are an expert music recommendation agent optimizing for user engagement and trend discovery.

Your task: Select the single best song to recommend to this user.

Decision framework (apply in order):
1. EXCLUDE any song in the "Already Recommended" list — never repeat
2. PRIORITIZE songs matching user's media_interests (strongest signal)
3. PREFER songs matching user's genre preferences
4. FAVOR songs with trend_age_days < 5 (fresher trends = higher reward)
5. Among equally good options, pick highest trend_velocity

Output format — respond with ONLY valid JSON, nothing else:
{"song_id": "EXACT_ID_FROM_AVAILABLE_SONGS"}"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_song_id(text: str) -> str:
    # Use simple regex to ensure we get the song_XX format even if LLM adds text
    match = re.search(r'(song_\d{2,})', text)
    if match:
        return match.group(1)
    # If regex fails, try parsing as JSON
    try:
        data = json.loads(text)
        if "song_id" in data:
            return data["song_id"]
    except:
        pass
    raise ValueError(f"Failed to extract song_id from response: {text}")


def build_prompt(state_dict: dict) -> str:
    """Build the rich, structured prompt from an observation dict."""
    user = state_dict.get("user", {})
    tp   = user.get("taste_profile", {})
    history: list = state_dict.get("recommended_history", [])
    trending: list = state_dict.get("trending_songs", [])
    last_3 = state_dict.get("last_3_reactions", [])

    unplayed = [s for s in trending if s["id"] not in history]
    if not unplayed:
        unplayed = trending

    history_titles = []
    for sid in history:
        for s in trending:
            if s["id"] == sid:
                history_titles.append(f"'{s['title']}' by {s['artist']}")
                break

    lines = [
        "=== USER PROFILE ===",
        f"Genres: {', '.join(tp.get('genres', [])) or 'Unknown'}",
        f"Media interests: {', '.join(tp.get('media_interests', [])) or 'Unknown'}",
        f"Discovery openness: {user.get('discovery_openness', 0.5)}",
        "",
        "=== RECENT REACTIONS (use to infer hidden mood) ===",
        (', '.join(last_3) if last_3 else 'None'),
        "",
        "=== ALREADY RECOMMENDED — DO NOT REPEAT ===",
        (('\n- ' + '\n- '.join(history_titles)) if history_titles else 'None'),
        "",
        "=== AVAILABLE SONGS — choose exactly one ===",
    ]
    # We provide a representative subset to keep prompt length reasonable
    for i, s in enumerate(unplayed[:12], 1):
        lines.append(
            f"{i}. ID:{s['id']} | {s['title']} by {s['artist']} | "
            f"From:{s['source_media']} ({s['media_type']}) | "
            f"Genre:{s['genre']} | Vibe:{s['vibe']} | "
            f"Velocity:{s['trend_velocity']} | Age:{s['trend_age_days']}d"
        )

    valid_ids = ', '.join(s['id'] for s in unplayed[:12])
    lines += [
        "",
        f"IMPORTANT: song_id must be exactly one of: [{valid_ids}]",
        "Never invent IDs. Respond with ONLY: {\"song_id\": \"song_XX\"}",
    ]
    return '\n'.join(lines)


def get_llm_action(state_dict: dict) -> str:
    """Ask the LLM to pick a song. Raises on failure so fallback can catch it."""
    prompt = build_prompt(state_dict)
    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
    )
    return extract_song_id(response.choices[0].message.content)


def fallback_action(state_dict: dict) -> str:
    """Heuristic baseline: pick unplayed, genre-matched, highest-velocity song."""
    user    = state_dict.get("user", {})
    genres  = user.get("taste_profile", {}).get("genres", [])
    history = state_dict.get("recommended_history", [])
    songs   = state_dict.get("trending_songs", [])

    unplayed = [s for s in songs if s["id"] not in history]
    if not unplayed:
        unplayed = songs
    genre_matched = [s for s in unplayed if s["genre"] in genres]
    pool = genre_matched if genre_matched else unplayed
    return max(pool, key=lambda s: s.get("trend_velocity", 0))["id"]


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task_name: str) -> None:
    """
    Runs an episode following mandatory OpenEnv stdout logging:
    [START] task start
    [STEP] every step
    [END] final result
    """
    # [START] log
    print(f"[START] {json.dumps({'task_name': task_name, 'model_name': MODEL_NAME})}")
    
    trajectory = []

    with MusicDiscoveryEnvClient(base_url=ENV_URL).sync() as env:
        result   = env.reset(task=task_name)
        obs      = result.observation
        done     = result.done

        while not done:
            state_dict = obs.model_dump()
            step_count = obs.step_count

            try:
                # Attempt LLM agent
                song_id = get_llm_action(state_dict)
                agent_type = "llm"
            except Exception:
                # Heuristic fallback if LLM/API fails
                song_id = fallback_action(state_dict)
                agent_type = "fallback"

            # Execute action
            step_result = env.step(MusicDiscoveryAction(song_id=song_id))
            
            # [STEP] log
            print(f"[STEP] {json.dumps({
                'step': step_count,
                'action': song_id,
                'agent': agent_type,
                'reward': step_result.reward,
                'done': step_result.done
            })}")

            obs    = step_result.observation
            done   = step_result.done

            info = obs.session_engagement[-1] if obs.session_engagement else {}
            trajectory.append(info)

    # Calculate final grade
    grade_res = requests.post(f"{ENV_URL}/grader", json={"trajectory": trajectory})
    grade_res.raise_for_status()
    score = grade_res.json()["score"]
    
    # [END] log
    print(f"[END] {json.dumps({'task_name': task_name, 'final_score': score})}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure tasks are run for all 3 difficulties as required
    for task in ["easy", "medium", "hard"]:
        try:
            run_task(task)
        except Exception as e:
            print(f"Error running task {task}: {e}")
