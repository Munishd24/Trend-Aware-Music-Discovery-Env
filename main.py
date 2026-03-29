from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
from models import Observation, Action, StepResult, GraderInput, GraderOutput, TaskConfig
from music_discovery_proto_v2 import MusicDiscoveryEnv, grade, baseline_agent

app = FastAPI(title="Music Discovery Environment")

# Global state
current_env = None

@app.post("/reset", response_model=Observation)
def reset_env(task: str = "easy"):
    global current_env
    if task not in ["easy", "medium", "hard"]:
        raise HTTPException(status_code=400, detail="Invalid task")
    current_env = MusicDiscoveryEnv(task_config=task)
    state = current_env.reset()
    return Observation(**state)

@app.post("/step", response_model=StepResult)
def step_env(action: Action):
    global current_env
    if not current_env:
        raise HTTPException(status_code=400, detail="Environment not reset")
        
    state, reward, done, info = current_env.step({"song_id": action.song_id})
    return StepResult(
        observation=Observation(**state),
        reward=reward,
        done=done,
        info=info
    )

@app.get("/state", response_model=Observation)
def get_state():
    global current_env
    if not current_env:
        raise HTTPException(status_code=400, detail="Environment not reset")
    return Observation(**current_env._get_state())

@app.get("/tasks", response_model=Dict[str, Any])
def get_tasks():
    tasks = [
        TaskConfig(name="easy", difficulty=0.2, description="User with strong anime interest, fresh trending songs"),
        TaskConfig(name="medium", difficulty=0.5, description="Mixed sources, aging trends, shifting user mood"),
        TaskConfig(name="hard", difficulty=0.8, description="Cold start user, diverse catalogue, varied trend ages")
    ]
    return {
        "tasks": [t.model_dump() for t in tasks],
        "action_schema": {"song_id": "string"}
    }

@app.post("/grader", response_model=GraderOutput)
def grade_endpoint(input_data: GraderInput):
    score = grade(input_data.trajectory)
    return GraderOutput(score=score)

@app.get("/baseline")
def run_baseline():
    import random
    random.seed(42)
    results = {}
    for task in ["easy", "medium", "hard"]:
        env = MusicDiscoveryEnv(task_config=task)
        state_dict = env.reset()
        traj = []
        done = False
        while not done:
            action_dict = baseline_agent(state_dict)
            state_dict, reward, done, info = env.step(action_dict)
            traj.append(info)
        results[task] = grade(traj)
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
