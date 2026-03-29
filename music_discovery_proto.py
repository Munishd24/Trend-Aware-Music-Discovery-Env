import json
import random

# --- Data Schemas & Environment Constants ---
GENRES = ["pop", "rock", "hip-hop", "electronic", "classical", "jazz"]
MEDIA_TYPES = ["anime", "game", "movie", "tv_show", "viral_video"]
MOODS = ["hyped", "relaxed", "focus", "party", "sad"]

class MusicDiscoveryEnv:
    def __init__(self, task_config="easy"):
        self.task_config = task_config
        self.max_steps = 10
        
    def _generate_song(self, force_media=None, age_range=(0, 5)):
        return {
            "id": f"song_{random.randint(1000, 9999)}",
            "title": f"Track {random.randint(10, 99)}",
            "artist": f"Artist {random.randint(1, 20)}",
            "source_media": f"{force_media or random.choice(MEDIA_TYPES)} media {random.randint(1, 100)}",
            "media_type": force_media if force_media else random.choice(MEDIA_TYPES),
            "trend_velocity": round(random.uniform(0.1, 1.0), 2),
            "trend_age_days": random.randint(*age_range),
            "genre": random.choice(GENRES),
            "vibe": random.choice(MOODS)
        }
        
    def reset(self):
        self.step_count = 0
        self.session_engagement = []
        
        # Configuration based on task
        if self.task_config == "easy":
            # user has strong anime interest, all 10 songs from anime sources, trend_age 1-3 days
            self.user = {
                "taste_profile": {"genres": ["pop", "electronic"], "media_interests": ["anime"]},
                "mood": "hyped",
                "discovery_openness": 0.8,
                "listening_history": []
            }
            self.trending_songs = [self._generate_song(force_media="anime", age_range=(1, 3)) for _ in range(10)]
            
        elif self.task_config == "medium":
            # mixed media sources, some songs 8-12 days old, mood shifts every 3 steps (handled in step)
            self.user = {
                "taste_profile": {"genres": ["rock", "hip-hop"], "media_interests": ["game", "movie"]},
                "mood": "party",
                "discovery_openness": 0.6,
                "listening_history": []
            }
            self.trending_songs = [self._generate_song(age_range=(1, 12)) for _ in range(15)]
            
        elif self.task_config == "hard":
            # cold start user (minimal history), 20 songs across all media types, varying trend ages
            self.user = {
                "taste_profile": {"genres": [], "media_interests": []},
                "mood": random.choice(MOODS),
                "discovery_openness": 0.5,
                "listening_history": []
            }
            self.trending_songs = [self._generate_song(age_range=(0, 20)) for _ in range(20)]
            
        return self._get_state()
        
    def _get_state(self):
        return {
            "user": self.user,
            "trending_songs": self.trending_songs,
            "step_count": self.step_count,
            "session_engagement": self.session_engagement
        }
        
    def step(self, action):
        song_id = action.get("song_id")
        song = next((s for s in self.trending_songs if s["id"] == song_id), None)
        
        if not song:
            # Invalid action penalty
            return self._get_state(), -1.0, True, {"error": "Invalid song_id"}
            
        if self.task_config == "medium" and self.step_count > 0 and self.step_count % 3 == 0:
            self.user["mood"] = random.choice(MOODS)
            
        self.step_count += 1
        
        # Simulate user reaction
        reaction, base_reward = self._simulate_reaction(song)
        
        # early discovery bonus: max(0.5, 1.0 - trend_age_days * 0.05)
        trend_age_bonus = max(0.5, 1.0 - song["trend_age_days"] * 0.05)
        
        taste_bonus = 0.0
        if song["genre"] in self.user["taste_profile"]["genres"]:
            taste_bonus += 0.2
        if song["vibe"] == self.user["mood"]:
            taste_bonus += 0.2
            
        final_reward = round(base_reward * trend_age_bonus + taste_bonus, 2)
        
        step_info = {
            "step": self.step_count,
            "song_id": song_id,
            "reaction": reaction,
            "reward": final_reward
        }
        self.session_engagement.append(step_info)
        self.user["listening_history"].append(song_id)
        
        done = self.step_count >= self.max_steps
        
        return self._get_state(), final_reward, done, step_info
        
    def _simulate_reaction(self, song):
        score = 0
        if song["genre"] in self.user["taste_profile"]["genres"]:
            score += 3
        if song["media_type"] in self.user["taste_profile"]["media_interests"]:
            score += 3
        if song["vibe"] == self.user["mood"]:
            score += 2
        
        # freshness gives a boost (0 to 2.5)
        score += max(0, 5 - song["trend_age_days"]) * 0.5
        
        if not self.user["taste_profile"]["genres"]:
            # Cold start logic: rely purely on randomness + openness
            score += random.random() * 5 * self.user["discovery_openness"]
            
        # Add some base randomness
        score += random.uniform(-1, 2)
            
        if score >= 6.5:
            return "shared", 1.0
        elif score >= 5:
            return "saved", 0.8
        elif score >= 3.5:
            return "added_to_playlist", 0.7
        elif score >= 1.5:
            return "played_once", 0.3
        else:
            return "skipped", -0.2

def grade(trajectory):
    if not trajectory:
        return 0.0
    total_reward = sum(step["reward"] for step in trajectory)
    # Estimate max possible reward logic for 10 steps.
    # Max base is 1.0. Max age bonus x1.0, plus 0.4 taste bonus = 1.4 per step -> 14.0 total.
    # Min possible might be negative. Let's floor it at 0.0
    max_possible = len(trajectory) * 1.4
    return min(1.0, max(0.0, total_reward / max_possible))

def baseline_agent(state):
    user = state["user"]
    genres = user["taste_profile"]["genres"]
    songs = state["trending_songs"]
    
    # Filter by user genre if available
    matching_songs = [s for s in songs if s["genre"] in genres] if genres else songs
    if not matching_songs:
        matching_songs = songs
        
    # Filter out already listened
    history = user["listening_history"]
    unplayed = [s for s in matching_songs if s["id"] not in history]
    if not unplayed:
        unplayed = songs # fallback to playing anything
        
    best_song = max(unplayed, key=lambda s: s["trend_velocity"])
    return {"song_id": best_song["id"]}

def demo():
    print("=== TREND-AWARE MUSIC DISCOVERY ENV DEMO ===")
    
    for task in ["easy", "medium", "hard"]:
        print(f"\\n--- Running Task: {task.upper()} ---")
        env = MusicDiscoveryEnv(task_config=task)
        state = env.reset()
        
        trajectory = []
        done = False
        total_reward = 0
        
        while not done:
            action = baseline_agent(state)
            state, reward, done, info = env.step(action)
            trajectory.append(info)
            total_reward += reward
            
        score = grade(trajectory)
        print(f"Steps taken: {env.step_count}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Final Grade (0.0 - 1.0): {score:.2f}")
        
    print("\\n--- Sample LLM Agent Interaction ---")
    env = MusicDiscoveryEnv(task_config="easy")
    state = env.reset()
    
    prompt = f"You are a Music Recommender AI.\\nUser Profile:\\n- Mood: {state['user']['mood']}\\n- Genres: {', '.join(state['user']['taste_profile']['genres'])}\\n- Interests: {', '.join(state['user']['taste_profile']['media_interests'])}\\n\\nAvailable Trending Songs:\\n"
    for s in state['trending_songs'][:3]:
        prompt += f"- ID: {s['id']} | {s['title']} by {s['artist']} ({s['genre']}, {s['vibe']}) | Trends: Velocity {s['trend_velocity']}, {s['trend_age_days']} days old\\n"
        
    print("\\n[Agent Prompt]")
    print(prompt)
    
    action_json = json.dumps({"song_id": state['trending_songs'][0]['id']}, indent=2)
    print("[Agent Response]")
    print(action_json)
    
    state, reward, done, info = env.step({"song_id": state['trending_songs'][0]['id']})
    print(f"\\n[Environment Output]")
    print(f"Action Result: {info['reaction']}")
    print(f"Reward Received: {reward}")

if __name__ == '__main__':
    demo()
