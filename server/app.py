# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Trend-Aware Music Discovery Environment.

Auto-generates /reset, /step, /state, /ws, /health, /web (Gradio) endpoints
via OpenEnv create_app(). Custom /tasks, /grader, /baseline added for hackathon.
"""

import random

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core"
    ) from e

try:
    from ..models import MusicDiscoveryAction, MusicDiscoveryObservation
    from .music_discovery_env_environment import MusicDiscoveryEnvironment, grade, baseline_agent
except (ImportError, SystemError):
    from models import MusicDiscoveryAction, MusicDiscoveryObservation
    from server.music_discovery_env_environment import MusicDiscoveryEnvironment, grade, baseline_agent

# Create the OpenEnv app (auto-generates /reset, /step, /state, /ws, /health, /web)
app = create_app(
    MusicDiscoveryEnvironment,
    MusicDiscoveryAction,
    MusicDiscoveryObservation,
    env_name="music_discovery_env",
    max_concurrent_envs=5,
)


# --- Custom Hackathon Endpoints ---

@app.get("/tasks")
def get_tasks():
    """List available task configurations."""
    return {
        "tasks": [
            {"name": "easy", "difficulty": 0.2, "description": "User with strong anime interest, fresh trending songs"},
            {"name": "medium", "difficulty": 0.5, "description": "Mixed sources, aging trends, shifting user mood (POMDP)"},
            {"name": "hard", "difficulty": 0.8, "description": "Cold start user, diverse catalogue, varied trend ages"},
        ],
        "action_schema": {"song_id": "string"},
    }


@app.post("/grader")
def grade_endpoint(input_data: dict):
    """Evaluate a completed trajectory and return a 0.0-1.0 score."""
    trajectory = input_data.get("trajectory", [])
    score = grade(trajectory)
    return {"score": score}


@app.get("/baseline")
def run_baseline():
    """Run the deterministic baseline agent across all tasks."""
    random.seed(42)
    results = {}
    for task in ["easy", "medium", "hard"]:
        env = MusicDiscoveryEnvironment()
        obs = env.reset(task=task)
        obs_dict = obs.model_dump()
        traj = []
        done = False
        while not done:
            action_dict = baseline_agent(obs_dict)
            obs = env.step(MusicDiscoveryAction(song_id=action_dict["song_id"]))
            obs_dict = obs.model_dump()
            step_info = obs_dict["session_engagement"][-1] if obs_dict["session_engagement"] else {}
            traj.append(step_info)
            done = obs.done
        results[task] = grade(traj)
    return results


# --- Mount custom Gradio UI ---
import os
if os.environ.get("ENABLE_WEB_INTERFACE", "").lower() == "true":
    try:
        import gradio as gr
        from gradio_ui import create_gradio_app
        gradio_demo = create_gradio_app()
        app = gr.mount_gradio_app(app, gradio_demo, path="/playground")
        print("✅ Gradio playground mounted at /playground")
    except Exception as e:
        print(f"⚠️  Gradio UI not loaded: {e}")


@app.get("/")
def root():
    """Root endpoint — redirect to the Gradio playground if available."""
    from fastapi.responses import RedirectResponse
    if os.environ.get("ENABLE_WEB_INTERFACE", "").lower() == "true":
        return RedirectResponse(url="/playground")
    return {
        "name": "Trend-Aware Music Discovery Environment",
        "docs": "/docs",
        "tasks": "/tasks",
        "baseline": "/baseline",
    }


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)

