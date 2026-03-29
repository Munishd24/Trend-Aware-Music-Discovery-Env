import os
import json
import requests
from openai import OpenAI
import time

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", HF_TOKEN or "dummy_key")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=API_BASE_URL
)

ENV_URL = "http://localhost:7860"

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
        
        prompt = f"""You are a highly capable Music Recommender AI.
User Profile:
- Mood: {user_profile['mood']}
- Genres: {', '.join(user_profile['taste_profile']['genres'])}
- Media Interests: {', '.join(user_profile['taste_profile']['media_interests'])}

Available Trending Songs:
"""
        for s in unplayed_trending[:10]:
            prompt += f"- ID: {s['id']} | '{s['title']}' by {s['artist']} (Trend Velocity: {s['trend_velocity']}, {s['trend_age_days']} days old, Genre: {s['genre']}, Vibe: {s['vibe']})\n"
            
        prompt += """\nRespond ONLY with a JSON object containing the song_id you recommend. Example: {"song_id": "song_01"}"""

        try:
            # Note: For hackathon offline/validation, mock or use minimal LLM call logic
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a direct JSON output agent. Recommend the best song for the user based on taste and trend age. Output ONLY JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            action_json = json.loads(response.choices[0].message.content)
            song_id = action_json.get("song_id")
            if not song_id:
                raise ValueError("No song_id in response")
        except Exception as e:
            user_genres = user_profile['taste_profile']['genres']
            history = state.get("recommended_history", [])
            valid_songs = [s for s in trending if s["id"] not in history]
            if not valid_songs: valid_songs = trending
            
            genre_songs = [s for s in valid_songs if s["genre"] in user_genres]
            if not genre_songs: genre_songs = valid_songs
            
            best_song = max(genre_songs, key=lambda s: s['trend_velocity'])
            song_id = best_song['id']
            print(f"LLM request skipped ({e}). Fallback selected unplayed, genre-matched trend: {song_id}")
            
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
