import json
import random

# --- Data Schemas & Environment Constants ---
GENRES = ["pop", "rock", "hip-hop", "electronic", "classical", "jazz"]
MEDIA_TYPES = ["anime", "game", "movie", "tv_show", "viral_video"]
MOOD_ROTATION = ["hyped", "relaxed", "party", "focus", "sad"]

REAL_SONGS_DB = [
    {"id": "song_01", "title": "Enemy", "artist": "Imagine Dragons", "source_media": "Arcane", "media_type": "tv_show", "genre": "pop", "vibe": "hyped"},
    {"id": "song_02", "title": "Chippin' In", "artist": "SAMURAI", "source_media": "Cyberpunk 2077", "media_type": "game", "genre": "rock", "vibe": "hyped"},
    {"id": "song_03", "title": "SPECIALZ", "artist": "King Gnu", "source_media": "Jujutsu Kaisen", "media_type": "anime", "genre": "rock", "vibe": "hyped"},
    {"id": "song_04", "title": "Idol", "artist": "YOASOBI", "source_media": "Oshi no Ko", "media_type": "anime", "genre": "pop", "vibe": "party"},
    {"id": "song_05", "title": "Bling-Bang-Bang-Born", "artist": "Creepy Nuts", "source_media": "Mashle", "media_type": "anime", "genre": "hip-hop", "vibe": "party"},
    {"id": "song_06", "title": "Night Dancer", "artist": "imase", "source_media": "TikTok", "media_type": "viral_video", "genre": "pop", "vibe": "relaxed"},
    {"id": "song_07", "title": "Cupid", "artist": "FIFTY FIFTY", "source_media": "TikTok", "media_type": "viral_video", "genre": "pop", "vibe": "relaxed"},
    {"id": "song_08", "title": "Running Up That Hill", "artist": "Kate Bush", "source_media": "Stranger Things", "media_type": "tv_show", "genre": "pop", "vibe": "focus"},
    {"id": "song_09", "title": "Master of Puppets", "artist": "Metallica", "source_media": "Stranger Things", "media_type": "tv_show", "genre": "rock", "vibe": "hyped"},
    {"id": "song_10", "title": "Peaches", "artist": "Jack Black", "source_media": "Super Mario Movie", "media_type": "movie", "genre": "pop", "vibe": "party"},
    {"id": "song_11", "title": "Goth", "artist": "Sidewalks and Skeletons", "source_media": "TikTok", "media_type": "viral_video", "genre": "electronic", "vibe": "focus"},
    {"id": "song_12", "title": "Makeba", "artist": "Jain", "source_media": "Levi's Ad", "media_type": "tv_show", "genre": "pop", "vibe": "party"},
    {"id": "song_13", "title": "Paint It, Black", "artist": "The Rolling Stones", "source_media": "Wednesday", "media_type": "tv_show", "genre": "rock", "vibe": "sad"},
    {"id": "song_14", "title": "Bloody Mary", "artist": "Lady Gaga", "source_media": "Wednesday", "media_type": "tv_show", "genre": "pop", "vibe": "party"},
    {"id": "song_15", "title": "Goo Goo Muck", "artist": "The Cramps", "source_media": "Wednesday", "media_type": "tv_show", "genre": "rock", "vibe": "party"},
    {"id": "song_16", "title": "Suzume", "artist": "RADWIMPS", "source_media": "Suzume", "media_type": "anime", "genre": "pop", "vibe": "sad"},
    {"id": "song_17", "title": "KICK BACK", "artist": "Kenshi Yonezu", "source_media": "Chainsaw Man", "media_type": "anime", "genre": "rock", "vibe": "hyped"},
    {"id": "song_18", "title": "Shikairo Days", "artist": "Shika-bu", "source_media": "My Deer Friend Nokotan", "media_type": "anime", "genre": "pop", "vibe": "party"},
    {"id": "song_19", "title": "The Last of Us", "artist": "Gustavo Santaolalla", "source_media": "The Last of Us", "media_type": "game", "genre": "classical", "vibe": "sad"},
    {"id": "song_20", "title": "Gwyn, Lord of Cinder", "artist": "Motoi Sakuraba", "source_media": "Dark Souls", "media_type": "game", "genre": "classical", "vibe": "sad"}
]

class MusicDiscoveryEnv:
    def __init__(self, task_config="easy"):
        self.task_config = task_config
        self.max_steps = 10
        
    def _get_trending_songs(self, force_media=None, age_range=(0, 5), count=10):
        songs = []
        candidates = [s for s in REAL_SONGS_DB if not force_media or s["media_type"] == force_media]
        if len(candidates) < count:
            candidates = REAL_SONGS_DB # Fallback if not enough candidates
            
        selected = random.sample(candidates, count)
        for s in selected:
            s_copy = dict(s)
            s_copy["trend_velocity"] = round(random.uniform(0.1, 1.0), 2)
            s_copy["trend_age_days"] = random.randint(*age_range)
            songs.append(s_copy)
        return songs
        
    def reset(self):
        self.step_count = 0
        self.session_engagement = []
        self.recommended_history = []
        
        # Configuration based on task
        if self.task_config == "easy":
            # user has strong anime interest, all 10 songs from anime sources, trend_age 1-3 days
            self.user = {
                "taste_profile": {"genres": ["pop", "rock"], "media_interests": ["anime"]},
                "mood": "hyped",
                "discovery_openness": 0.8,
                "listening_history": []
            }
            self.trending_songs = self._get_trending_songs(force_media="anime", age_range=(1, 3), count=10)
            
        elif self.task_config == "medium":
            # mixed media sources, some songs 8-12 days old, mood shifts every 3 steps
            self.user = {
                "taste_profile": {"genres": ["rock", "pop"], "media_interests": ["game", "tv_show"]},
                "mood": MOOD_ROTATION[0],
                "discovery_openness": 0.6,
                "listening_history": []
            }
            self.trending_songs = self._get_trending_songs(age_range=(3, 15), count=15)
            
        elif self.task_config == "hard":
            # cold start user (minimal history, 2 seed songs), 20 songs across all media types, varying trend ages
            seed_songs = random.sample([s["id"] for s in REAL_SONGS_DB], 2)
            self.user = {
                "taste_profile": {"genres": [], "media_interests": []},
                "mood": random.choice(MOOD_ROTATION),
                "discovery_openness": 0.5,
                "listening_history": seed_songs
            }
            self.trending_songs = self._get_trending_songs(age_range=(0, 20), count=20)
            
        return self._get_state()
        
    def _get_state(self):
        return {
            "user": self.user,
            "trending_songs": self.trending_songs,
            "step_count": self.step_count,
            "session_engagement": self.session_engagement,
            "recommended_history": self.recommended_history
        }
        
    def step(self, action):
        song_id = action.get("song_id")
        song = next((s for s in self.trending_songs if s["id"] == song_id), None)
        
        if not song:
            return self._get_state(), -1.0, True, {"error": "Invalid song_id"}
            
        if self.task_config == "medium" and self.step_count > 0 and self.step_count % 3 == 0:
            current_mood_idx = MOOD_ROTATION.index(self.user["mood"])
            self.user["mood"] = MOOD_ROTATION[(current_mood_idx + 1) % len(MOOD_ROTATION)]
            
        self.step_count += 1
        
        if song_id in self.recommended_history:
            step_info = {
                "step": self.step_count,
                "song_id": song_id,
                "reaction": "ignored",
                "reward": -0.3,
                "repeated": True,
                "trend_age_days": song["trend_age_days"]
            }
            self.session_engagement.append(step_info)
            done = self.step_count >= self.max_steps
            return self._get_state(), -0.3, done, step_info
            
        self.recommended_history.append(song_id)
        
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
            "reward": final_reward,
            "trend_age_days": song["trend_age_days"]
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
        
        score += max(0, 5 - song["trend_age_days"]) * 0.5
        
        if not self.user["taste_profile"]["genres"]:
            score += random.random() * 5 * self.user["discovery_openness"]
            
        # Adjust base randomness slightly downwards to tighten everything
        score += random.uniform(-1.5, 1.0)
            
        if self.task_config == "hard":
            t_shared, t_saved, t_playlist, t_played = 9.0, 7.5, 5.0, 2.0
        elif self.task_config == "medium":
            t_shared, t_saved, t_playlist, t_played = 8.5, 7.0, 5.5, 3.5
        else:
            t_shared, t_saved, t_playlist, t_played = 6.5, 5.0, 3.5, 1.5
            
        if score >= t_shared: return "shared", 1.0
        elif score >= t_saved: return "saved", 0.8
        elif score >= t_playlist: return "added_to_playlist", 0.7
        elif score >= t_played: return "played_once", 0.3
        else: return "skipped", -0.2

def grade(trajectory):
    if not trajectory:
        return 0.0
    positive_steps = [s for s in trajectory if s["reward"] > 0]
    engagement_rate = len(positive_steps) / len(trajectory)
    avg_reward = sum(s["reward"] for s in trajectory) / len(trajectory)
    discovery_bonus = sum(
        1 for s in trajectory 
        if s.get("trend_age_days", 10) < 3 and s["reaction"] in ["shared", "saved"]
    ) / len(trajectory)
    return min(1.0, max(0.0, (engagement_rate * 0.4) + (avg_reward * 0.4) + (discovery_bonus * 0.2)))

def baseline_agent(state):
    user = state["user"]
    genres = user["taste_profile"]["genres"]
    songs = state["trending_songs"]
    
    # Filter by user genre if available
    matching_songs = [s for s in songs if s["genre"] in genres] if genres else songs
    if not matching_songs:
        matching_songs = songs
        
    history = user["listening_history"]
    unplayed = [s for s in matching_songs if s["id"] not in history]
    if not unplayed:
        unplayed = songs
        
    best_song = max(unplayed, key=lambda s: s["trend_velocity"])
    return {"song_id": best_song["id"]}

def demo():
    random.seed(42) # Seed for reproducibility in the demo run
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

if __name__ == '__main__':
    demo()
