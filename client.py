# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Music Discovery Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import MusicDiscoveryAction, MusicDiscoveryObservation, UserProfile, TasteProfile


class MusicDiscoveryEnvClient(
    EnvClient[MusicDiscoveryAction, MusicDiscoveryObservation, State]
):
    """
    Client for the Trend-Aware Music Discovery Environment.

    Example:
        >>> with MusicDiscoveryEnvClient(base_url="http://localhost:7860").sync() as client:
        ...     obs = client.reset()
        ...     result = client.step(MusicDiscoveryAction(song_id="song_03"))
        ...     print(result.observation.last_3_reactions)
    """

    def _step_payload(self, action: MusicDiscoveryAction) -> Dict:
        return {"song_id": action.song_id}

    def _parse_result(self, payload: Dict) -> StepResult[MusicDiscoveryObservation]:
        obs_data = payload.get("observation", {})
        user_data = obs_data.get("user", {})
        tp = user_data.get("taste_profile", {})

        observation = MusicDiscoveryObservation(
            user=UserProfile(
                taste_profile=TasteProfile(
                    genres=tp.get("genres", []),
                    media_interests=tp.get("media_interests", []),
                ),
                discovery_openness=user_data.get("discovery_openness", 0.5),
                listening_history=user_data.get("listening_history", []),
            ),
            trending_songs=obs_data.get("trending_songs", []),
            step_count=obs_data.get("step_count", 0),
            session_engagement=obs_data.get("session_engagement", []),
            recommended_history=obs_data.get("recommended_history", []),
            last_3_reactions=obs_data.get("last_3_reactions", []),
            global_mood_trend=obs_data.get("global_mood_trend", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
