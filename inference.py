import os
import json
import re
import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", HF_TOKEN or "dummy_key")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=API_BASE_URL
)

ENV_URL = "http://localhost:7860"

SYSTEM_PROMPT = """You are an expert music recommendation agent optimizing for user engagement and trend discovery.

Your task: Select the single best song to recommend to this user.

Decision framework (apply in order):
1. EXCLUDE any song in recommended_history - never repeat
2. PRIORITIZE songs matching user's media_interests (strongest signal)
3. PREFER songs matching user's mood and genre preferences
4. FAVOR songs with trend_age_days < 5 (fresher trends = higher reward)
5. Among equally good options, pick highest trend_velocity

Output format - respond with ONLY this JSON, nothing else:
{"song_id": "EXACT_ID_FROM_AVAILABLE_SONGS"}"""

def extract_song_id(response_text: str) -> str:
    # 1. Try direct json.loads()
    try:
        data = json.loads(response_text)
        if "song_id" in data:
            return data["song_id"]
    except json.JSONDecodeError:
        pass
        
    # 2. Try regex for exact JSON block
    match = re.search(r'\{.*?"song_id"\s*:\s*"([^"]+)".*?\}', response_text, re.DOTALL)
    if match:
        return match.group(1)
        
    # 3. Extract any song_XX pattern
    fallback_match = re.search(r'(song_\d{2,})', response_text)
    if fallback_match:
        return fallback_match.group(1)
        
    raise ValueError("Failed to extract song_id from response")

def run_task(task_name):
    import random
    random.seed(42)
    print(f"\n--- Running Task: {task_name.upper()} ---")
    res = requests.post(f"{ENV_URL}/reset?task={task_name}")
    res.raise_for_status()
    state = res.json()
    
    trajectory = []
    done = False
    
    while not done:
        user_profile = state['user']
        trending = state['trending_songs']
        
        history = state.get('recommended_history', [])
        unplayed_trending = [s for s in trending if s['id'] not in history]
        
        history_titles = []
        for sid in history:
            for t_song in trending:
                if t_song['id'] == sid:
                    history_titles.append(f"'{t_song['title']}' by {t_song['artist']}")
                    break
        
        history_str = "\n- ".join(history_titles) if history_titles else "None"
        
        prompt = f"""=== USER PROFILE ===
Mood: {user_profile['mood']}
Favourite genres: {', '.join(user_profile['taste_profile']['genres'])}
Media interests: {', '.join(user_profile['taste_profile']['media_interests'])}
Discovery openness: {user_profile['discovery_openness']}

=== ALREADY RECOMMENDED (DO NOT PICK THESE) ===
{history_str}

=== AVAILABLE SONGS (choose exactly one) ===
"""
        for i, s in enumerate(unplayed_trending[:10], 1):
            prompt += f"{i}. ID: {s['id']} | Title: {s['title']} | Artist: {s['artist']} | From: {s['source_media']} ({s['media_type']}) | Genre: {s['genre']} | Vibe: {s['vibe']} | Trend velocity: {s['trend_velocity']} | Days trending: {s['trend_age_days']}\n"
            
        prompt += """
=== YOUR TASK ===
Apply the decision framework above.
Respond with ONLY: {"song_id": "song_XX"}"""

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            response_text = response.choices[0].message.content
            song_id = extract_song_id(response_text)
            
        except Exception as e:
            user_genres = user_profile['taste_profile']['genres']
            valid_songs = unplayed_trending if unplayed_trending else trending
            genre_songs = [s for s in valid_songs if s["genre"] in user_genres]
            if not genre_songs: genre_songs = valid_songs
            
            best_song = max(genre_songs, key=lambda s: s['trend_velocity'])
            song_id = best_song['id']
            print(f"LLM request skipped/failed ({e}). Fallback selected unplayed, genre-matched trend: {song_id}")
            
        step_res = requests.post(f"{ENV_URL}/step", json={"song_id": song_id})
        step_res.raise_for_status()
        step_data = step_res.json()
        
        state = step_data['observation']
        reward = step_data['reward']
        done = step_data['done']
        info = step_data['info']
        
        trajectory.append(info)
        print(f"Step {state['step_count']}: Action {song_id} -> Reaction: {info.get('reaction')}, Reward: {reward}")

    grade_res = requests.post(f"{ENV_URL}/grader", json={"trajectory": trajectory})
    grade_res.raise_for_status()
    final_score = grade_res.json()["score"]
    
    print(f"Final Grade for {task_name}: {final_score:.2f}")

if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)
