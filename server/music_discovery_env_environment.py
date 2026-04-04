# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Trend-Aware Music Discovery Environment Implementation.

An RL environment where an LLM agent acts as a music recommender,
optimizing for user engagement and trend freshness across 3 difficulty levels.
"""

import json
import random
from pathlib import Path
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MusicDiscoveryAction, MusicDiscoveryObservation, UserProfile, TasteProfile
except ImportError:
    from models import MusicDiscoveryAction, MusicDiscoveryObservation, UserProfile, TasteProfile

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
    {"id": "song_20", "title": "Gwyn, Lord of Cinder", "artist": "Motoi Sakuraba", "source_media": "Dark Souls", "media_type": "game", "genre": "classical", "vibe": "sad"},
]


class MusicDiscoveryEnvironment(Environment):
    """
    Trend-Aware Music Discovery RL Environment.

    An LLM agent acts as a music recommender, selecting songs from a trending
    catalog to maximize user engagement. Supports 3 difficulty levels:
    easy (echo chamber), medium (POMDP with mood shifts), hard (cold start).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    full_catalog = None

    def __init__(self):
        super().__init__()
        if MusicDiscoveryEnvironment.full_catalog is None:
            catalog_path = Path(__file__).parent / "catalog.json"
            if catalog_path.exists():
                with open(catalog_path, "r", encoding="utf-8") as f:
                    MusicDiscoveryEnvironment.full_catalog = json.load(f)
            else:
                MusicDiscoveryEnvironment.full_catalog = REAL_SONGS_DB
                
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task_config = "easy"
        self.max_steps = 10
        self._user = {}
        self._trending_songs = []
        self._session_engagement = []
        self._recommended_history = []
        self._global_mood_trend = ""
        self._session_genres = []
        self._exploration_budget = 2

    def _get_trending_songs(self, force_media=None, age_range=(0, 5), count=10):
        catalog = self.full_catalog if self.full_catalog else REAL_SONGS_DB
        selected = random.sample(catalog, count)
        songs = []
        for s in selected:
            s_copy = dict(s)
            
            if force_media:
                s_copy["media_type"] = force_media
                s_copy["source_media"] = "Anime" if force_media == "anime" else force_media.title()
            else:
                s_copy["source_media"] = random.choice(["TikTok", "Anime", "Movie", "Game", "TV Show"])
                s_copy["media_type"] = s_copy["source_media"].lower().replace(" ", "_")
                
            if "trend_velocity" not in s_copy:
                s_copy["trend_velocity"] = round(random.uniform(0.1, 1.0), 2)
                
            if s_copy.get("vibe") == getattr(self, "_global_mood_trend", ""):
                s_copy["trend_velocity"] = min(1.0, round(s_copy["trend_velocity"] + 0.5, 2))
                
            s_copy["trend_age_days"] = random.randint(*age_range)
            songs.append(s_copy)
        return songs

    def reset(self, task: str = "easy", **kwargs) -> MusicDiscoveryObservation:  # type: ignore[override]
        """Reset the environment for a specific task difficulty."""
        self.task_config = task
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._session_engagement = []
        self._recommended_history = []
        self._global_mood_trend = random.choice(MOOD_ROTATION)
        self._session_genres = []
        self._exploration_budget = 2

        if self.task_config == "easy":
            self._user = {
                "taste_profile": {"genres": ["pop", "rock"], "media_interests": ["anime"]},
                "mood": "hyped", "discovery_openness": 0.8, "listening_history": [],
            }
            self._trending_songs = self._get_trending_songs(force_media="anime", age_range=(1, 3), count=10)
        elif self.task_config == "medium":
            self._user = {
                "taste_profile": {"genres": ["rock", "pop"], "media_interests": ["game", "tv_show"]},
                "mood": MOOD_ROTATION[0], "discovery_openness": 0.6, "listening_history": [],
            }
            self._trending_songs = self._get_trending_songs(age_range=(3, 15), count=15)
        elif self.task_config == "hard":
            catalog = self.full_catalog if self.full_catalog else REAL_SONGS_DB
            seed_songs = random.sample([s["id"] for s in catalog], 2)
            self._user = {
                "taste_profile": {"genres": [], "media_interests": []},
                "mood": random.choice(MOOD_ROTATION), "discovery_openness": 0.45,
                "listening_history": seed_songs,
            }
            self._trending_songs = self._get_trending_songs(age_range=(0, 20), count=20)

        return self._build_observation()

    def step(self, action: MusicDiscoveryAction, **kwargs) -> MusicDiscoveryObservation:  # type: ignore[override]
        """Execute a recommendation action and return the user's reaction."""
        song_id = action.song_id
        song = next((s for s in self._trending_songs if s["id"] == song_id), None)

        if not song:
            return self._build_observation(reward=-1.0, done=True)

        if self.task_config == "medium" and self._state.step_count > 0:
            if random.random() < 0.25:
                idx = MOOD_ROTATION.index(self._user["mood"])
                self._user["mood"] = MOOD_ROTATION[(idx + 1) % len(MOOD_ROTATION)]

        self._state.step_count += 1

        if song_id in self._recommended_history:
            self._session_engagement.append({
                "step": self._state.step_count, "song_id": song_id,
                "reaction": "ignored", "reward": -0.3, "repeated": True,
                "trend_age_days": song["trend_age_days"],
            })
            done = self._state.step_count >= self.max_steps
            return self._build_observation(reward=-0.3, done=done)

        self._recommended_history.append(song_id)
        reaction, base_reward = self._simulate_reaction(song)
        trend_age_bonus = max(0.5, 1.0 - song["trend_age_days"] * 0.05)

        taste_bonus = 0.0
        if song["genre"] in self._user["taste_profile"]["genres"]:
            taste_bonus += 0.2
        if song["vibe"] == self._user["mood"]:
            taste_bonus += 0.2
            
        if song.get("vibe") == self._user["mood"] == getattr(self, "_global_mood_trend", ""):
            taste_bonus += 1.0

        # Implement new Diversity Logic
        diversity_bonus = 0.0
        if getattr(self, "_exploration_budget", 0) > 0 and song.get("genre") not in getattr(self, "_session_genres", []):
            self._exploration_budget -= 1
            if reaction in ["shared", "saved", "added_to_playlist"]:
                diversity_bonus += 1.0
                
        if song.get("genre") not in getattr(self, "_session_genres", []):
            self._session_genres.append(song.get("genre"))

        raw_reward = round(base_reward * trend_age_bonus + taste_bonus + diversity_bonus, 2)
        final_reward = max(-1.0, min(1.0, raw_reward))

        self._session_engagement.append({
            "step": self._state.step_count, "song_id": song_id,
            "reaction": reaction, "reward": final_reward,
            "trend_age_days": song["trend_age_days"],
        })
        self._user["listening_history"].append(song_id)

        done = self._state.step_count >= self.max_steps
        return self._build_observation(reward=final_reward, done=done)

    def _simulate_reaction(self, song):
        score = 0
        if song["genre"] in self._user["taste_profile"]["genres"]:
            score += 3
        if song["media_type"] in self._user["taste_profile"]["media_interests"]:
            score += 3
        if song["vibe"] == self._user["mood"]:
            score += 2
            
        if song["vibe"] == self._user["mood"] == getattr(self, "_global_mood_trend", ""):
            score += 4
            self._user["discovery_openness"] = min(1.0, self._user["discovery_openness"] + 0.15)
            
        score += max(0, 5 - song["trend_age_days"]) * 0.5

        if not self._user["taste_profile"]["genres"]:
            score += random.random() * 5 * self._user["discovery_openness"]

        if self.task_config == "easy":
            noise_lo, noise_hi = -0.3, 0.3
        elif self.task_config == "medium":
            noise_lo, noise_hi = -0.8, 0.8
        else:
            noise_lo, noise_hi = -0.5, 0.5
        score += random.uniform(noise_lo, noise_hi)

        if self.task_config == "hard":
            t_shared, t_saved, t_playlist, t_played = 7.5, 5.5, 3.5, 1.5
        elif self.task_config == "medium":
            t_shared, t_saved, t_playlist, t_played = 8.5, 7.0, 5.5, 3.5
        else:
            t_shared, t_saved, t_playlist, t_played = 6.5, 5.0, 3.5, 1.5

        if score >= t_shared: return "shared", 1.0
        elif score >= t_saved: return "saved", 0.8
        elif score >= t_playlist: return "added_to_playlist", 0.7
        elif score >= t_played: return "played_once", 0.3
        else: return "skipped", -0.2

    @property
    def state(self) -> State:
        return self._state

    def _build_observation(self, reward=0.0, done=False):
        tp = self._user.get("taste_profile", {"genres": [], "media_interests": []})
        last_3 = [s["reaction"] for s in self._session_engagement[-3:]]

        return MusicDiscoveryObservation(
            user=UserProfile(
                taste_profile=TasteProfile(genres=tp["genres"], media_interests=tp["media_interests"]),
                discovery_openness=self._user.get("discovery_openness", 0.5),
                listening_history=self._user.get("listening_history", []),
            ),
            trending_songs=self._trending_songs,
            step_count=self._state.step_count,
            session_engagement=self._session_engagement,
            recommended_history=self._recommended_history,
            last_3_reactions=last_3,
            global_mood_trend=getattr(self, "_global_mood_trend", ""),
            session_genres=getattr(self, "_session_genres", []),
            exploration_budget=getattr(self, "_exploration_budget", 0),
            done=done,
            reward=reward,
        )


def grade(trajectory):
    if not trajectory:
        return 0.0
    positive_steps = [s for s in trajectory if s.get("reward", 0) > 0]
    engagement_rate = len(positive_steps) / len(trajectory)
    avg_reward = sum(s.get("reward", 0) for s in trajectory) / len(trajectory)
    discovery_bonus = sum(
        1 for s in trajectory
        if s.get("trend_age_days", 10) < 3 and s.get("reaction") in ["shared", "saved"]
    ) / len(trajectory)
    return min(1.0, max(0.0, (engagement_rate * 0.4) + (avg_reward * 0.4) + (discovery_bonus * 0.2)))


def baseline_agent(state_dict):
    user = state_dict if "taste_profile" in state_dict else state_dict.get("user", state_dict)
    tp = user.get("taste_profile", {})
    genres = tp.get("genres", [])
    media = tp.get("media_interests", [])
    songs = state_dict.get("trending_songs", [])
    
    history = state_dict.get("recommended_history", [])
    if not history:
        history = user.get("listening_history", [])
        
    session_genres = state_dict.get("session_genres", [])
    exploration_budget = state_dict.get("exploration_budget", 0)
        
    unplayed = [s for s in songs if s["id"] not in history]
    if not unplayed:
        unplayed = songs

    # Epsilon-greedy exploration
    if exploration_budget > 0 and random.random() < 0.2:
        new_genre_songs = [s for s in unplayed if s.get("genre") not in session_genres]
        if new_genre_songs:
            best = max(new_genre_songs, key=lambda s: s.get("trend_velocity", 0.0))
            return {"song_id": best["id"]}

    def heuristic_score(s):
        score_val = 0
        if s.get("genre") in genres:
            score_val += 3.0
        if s.get("media_type") in media:
            score_val += 3.0
        if s.get("vibe") == state_dict.get("global_mood_trend", ""):
            score_val += 2.0
            
        # Age penalty
        score_val -= s.get("trend_age_days", 5) * 0.5
        # Velocity bonus
        score_val += s.get("trend_velocity", 0.0) * 2.0
        return score_val

    best = max(unplayed, key=heuristic_score)
    return {"song_id": best["id"]}

