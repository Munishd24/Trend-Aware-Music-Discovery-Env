from pydantic import BaseModel
from typing import List, Dict, Any

class Song(BaseModel):
    id: str
    title: str
    artist: str
    source_media: str
    media_type: str
    trend_velocity: float
    trend_age_days: int
    genre: str
    vibe: str

class TasteProfile(BaseModel):
    genres: List[str]
    media_interests: List[str]

class UserProfile(BaseModel):
    taste_profile: TasteProfile
    mood: str
    discovery_openness: float
    listening_history: List[str]

class Observation(BaseModel):
    user: UserProfile
    trending_songs: List[Song]
    step_count: int
    session_engagement: List[Dict[str, Any]]
    recommended_history: List[str]

class Action(BaseModel):
    song_id: str

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]

class GraderInput(BaseModel):
    trajectory: List[Dict[str, Any]]

class GraderOutput(BaseModel):
    score: float

class TaskConfig(BaseModel):
    name: str
    difficulty: float
    description: str
