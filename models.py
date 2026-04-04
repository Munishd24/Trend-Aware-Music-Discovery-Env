# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Trend-Aware Music Discovery Environment.
"""

from typing import List, Dict, Any, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class Song(Observation):
    """A trending song available for recommendation."""
    id: str = ""
    title: str = ""
    artist: str = ""
    source_media: str = ""
    media_type: str = ""
    trend_velocity: float = 0.0
    trend_age_days: int = 0
    genre: str = ""
    vibe: str = ""


class TasteProfile(Observation):
    """User's taste preferences."""
    genres: List[str] = Field(default_factory=list)
    media_interests: List[str] = Field(default_factory=list)


class UserProfile(Observation):
    """User profile visible to the agent (mood is hidden for POMDP)."""
    taste_profile: TasteProfile = Field(default_factory=TasteProfile)
    discovery_openness: float = 0.5
    listening_history: List[str] = Field(default_factory=list)


class MusicDiscoveryAction(Action):
    """Action for the Music Discovery environment — select a song to recommend."""
    song_id: str = Field(..., description="ID of the song to recommend (e.g., 'song_01')")


class MusicDiscoveryObservation(Observation):
    """Observation from the Music Discovery environment."""
    user: Optional[UserProfile] = None
    trending_songs: List[Dict[str, Any]] = Field(default_factory=list)
    step_count: int = 0
    session_engagement: List[Dict[str, Any]] = Field(default_factory=list)
    recommended_history: List[str] = Field(default_factory=list)
    last_3_reactions: List[str] = Field(default_factory=list)
    global_mood_trend: str = ""
    session_genres: List[str] = Field(default_factory=list)
    exploration_budget: int = 2
